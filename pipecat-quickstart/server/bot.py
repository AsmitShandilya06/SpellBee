import asyncio
import os
from enum import Enum, auto

from dotenv import load_dotenv
from groq import AsyncGroq
from loguru import logger

from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame, Frame, TranscriptionFrame,
    OutputTransportMessageFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame, CancelFrame, TTSSpeakFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

WORD_LIST = [
    "cat", "dog", "apple", "bridge", "cloud",
    "dance", "earth", "flame", "grace", "happy",
]

class GameState(Enum):
    IDLE        = auto()
    SPEAKING    = auto()    
    LISTENING   = auto()    
    WARNING     = auto()    
    EVALUATING  = auto()    
    GAME_OVER   = auto()

class SpellBeeProcessor(FrameProcessor):
    def __init__(self, word_list: list):
        super().__init__()
        self.word_list = word_list
        self.current_idx = 0
        self.score = 0
        self.state = GameState.IDLE
        self.bot_speaking = False
        self.interrupted = False  # Track if the current speech was cut off

        self.llm = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    async def _speak_text(self, text: str):
        await self.push_frame(TTSSpeakFrame(text))

    async def start_game(self):
        logger.info("SpellBeeProcessor: game starting")
        self.current_idx = 0
        self.score = 0
        self.state = GameState.SPEAKING
        self.bot_speaking = False
        self.interrupted = False
        
        word = self.word_list[self.current_idx]
        intro = (
            f"Welcome to Spell Bee! I will say a word and you spell it letter by letter. "
            f"Ready? Your first word is: {word}. Please spell {word}."
        )
        await self._speak_text(intro)
        await self._push_ui_update()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            if self.bot_speaking:
                logger.info("User interrupted bot! Canceling output...")
               
                await self.push_frame(CancelFrame())
            
                self.interrupted = True
                self.bot_speaking = False 
                
                if self.state == GameState.SPEAKING:
                    self.state = GameState.WARNING
            return

       
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
            self.interrupted = False
            logger.info("Bot started speaking")
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_speaking = False
            logger.info("Bot stopped speaking")

            if self.interrupted:
                self.interrupted = False
                return

            if self.state == GameState.SPEAKING:
                self.state = GameState.LISTENING
                logger.info("SpellBeeProcessor: LISTENING for user spelling")
                await self._push_ui_update()

            elif self.state == GameState.WARNING:
                word = self.word_list[self.current_idx]
                self.state = GameState.SPEAKING
                await self._speak_text(f"Now, let's try again. Please spell {word}.")
                await self._push_ui_update()

            elif self.state == GameState.GAME_OVER:
                await self.push_frame(EndFrame())
            return

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return

            # If we are in Speaking state, the user interrupted us, and this text is WHAT they interrupted with.
            if self.state == GameState.SPEAKING or self.state == GameState.WARNING:
                logger.info(f"Received interruption text: '{text}'")
                # Do NOT transition state here. Let the scold finish, which will transition to SPEAKING.
                asyncio.create_task(self._handle_interruption(text))
                
            elif self.state == GameState.LISTENING:
                logger.info(f"Received spelling text: '{text}'")
                asyncio.create_task(self._evaluate_spelling(text))
            
            else:
                logger.debug(f"Ignoring transcription in state {self.state.name}: '{text}'")
            return

        await self.push_frame(frame, direction)

    async def _handle_interruption(self, user_text: str):
        # Pause briefly to ensure CancelFrame has flushed the TTS buffers
        await asyncio.sleep(0.5) 
        
        logger.info(f"Processing interruption via Groq for text: '{user_text}'")
        system_prompt = (
            "You are an AI hosting a Spell Bee. The user interrupted you while you were speaking. "
            f"They said: '{user_text}'. "
            "Understand their intent, but politely tell them to listen to the word carefully "
            "and let you complete speaking before they answer. Keep your response to exactly one brief sentence."
        )
        
        try:
            response = await self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}]
            )
            reply = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            reply = "Please wait until I finish speaking before you answer."
            
        await self._speak_text(reply)
        await self._push_ui_update()

    async def _evaluate_spelling(self, user_text: str):
        self.state = GameState.EVALUATING
        target = self.word_list[self.current_idx]
        logger.info(f"Using Groq to extract spelling from: '{user_text}'")
        
        system_prompt = (
            "You are a spelling evaluator. The user is trying to dictate the spelling of a word. "
            f"Extract ONLY the sequence of letters they dictated from this speech: '{user_text}'. "
            "Example 1: 'Umm I think it is c a t' -> 'cat' \n"
            "Example 2: 'a p p l e' -> 'apple' \n"
            "Example 3: 'apple' -> 'apple' \n"
            "Return ONLY the letters, all lowercase, no spaces, no punctuation. "
            "If they did not dictate any letters or words, return the word 'invalid'."
        )
        
        try:
            response = await self.llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}]
            )
            extracted = response.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            extracted = "invalid"

        logger.info(f"Groq extracted: '{extracted}' (Target: '{target}')")

        if extracted == "invalid":
            reply = f"I didn't quite catch a spelling there. Please spell the word: {target}."
            self.state = GameState.SPEAKING
            await self._speak_text(reply)
            return

        correct = (extracted == target.lower())
        if correct:
            self.score += 1
            reply = "Correct! Well done. "
        else:
            spelled_out = " ".join(target.upper())
            reply = f"Not quite. The correct spelling is {spelled_out}. "

        self.current_idx += 1
        if self.current_idx >= len(self.word_list):
            self.state = GameState.GAME_OVER
            reply += f"That is all the words! Your final score is {self.score} out of {len(self.word_list)}. Great effort!"
        else:
            next_word = self.word_list[self.current_idx]
            reply += f"Your next word is: {next_word}. Please spell {next_word}."
            self.state = GameState.SPEAKING

        await self._speak_text(reply)
        await self._push_ui_update()

    async def _push_ui_update(self):
        payload = {
            "type": "SPELL_BEE_UPDATE",
            "state": self.state.name,
            "score": self.score,
            "current_word_idx": self.current_idx,
            "total_words": len(self.word_list),
        }
        await self.push_frame(OutputTransportMessageFrame(message=payload))

async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info("Starting Spell Bee bot for new connection")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(voice="aura-helios-en")
    )
    spell_bee = SpellBeeProcessor(word_list=WORD_LIST)

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.8,
                start_secs=0.2,
                confidence=0.7,
                min_volume=0.6,
            )
        )
    )

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        spell_bee,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(transport, client):
        logger.info("Client connected — waiting for START_GAME message")

    @transport.event_handler("on_app_message")
    async def _on_app_message(transport, message, sender):
        logger.info(f"App message from {sender}: {message}")
        if isinstance(message, dict) and message.get("type") == "START_GAME":
            await spell_bee.start_game()

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

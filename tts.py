from gtts import gTTS
from io import BytesIO
from mpg123 import Mpg123, Out123
from speechkit import Session, SpeechSynthesis
import simpleaudio as sa
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

oauth_token = config['yandex_cloud']['oauth_token']
catalog_id = config['yandex_cloud']['catalog_id']
voice = config['yandex_cloud']['voice']
emotion = config['yandex_cloud']['emotion']
session = Session.from_yandex_passport_oauth_token(oauth_token, catalog_id)
synthesizeAudio = SpeechSynthesis(session)

def say_text_yandex(text): 
    sample_rate = 16000 # частота дискретизации должна 
                    # совпадать при синтезе и воспроизведении
    audio_data = synthesizeAudio.synthesize_stream(
        text=text,
        voice=voice, emotion=emotion, format='lpcm', sampleRateHertz=sample_rate)
    # Воспроизводим синтезированный файл
    play_obj = sa.play_buffer(
	    audio_data, # audio_data, полученная методом `.synthesize_stream()`
        1, # монодорожка, один канал
        2, # Количество байтов в секунду (16 bit = 2 bytes)
        sample_rate, # такой же как указали при запросе (8000, 16000, 48000)
        )
    play_obj.wait_done()

def sayText(text, lang='en'):
    if lang == 'ru':
        try:
            say_text_yandex(text)
        except Exception:
            print('Error using Yandex TTS')
            pass
        return
    try:
        mp3_fp = BytesIO()
        tts = gTTS(text, lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        mp3 = Mpg123()
        mp3.feed(mp3_fp.read())
        out = Out123()
        for frame in mp3.iter_frames(out.start):
            out.play(frame)
    except Exception:
        print('Error using Google TTS')
        pass


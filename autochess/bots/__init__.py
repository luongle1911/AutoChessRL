"""Bot luật.

BOTS          bot đọc view (một ván, cho lớp chơi thử): act(view, mask, rng) -> int
ARRAY_BOTS    bot luật vector hóa (bots/scripted.py): tempo, hyper, roll7, fast9
"""
from . import random_bot, econ_bot
from .scripted import PRESETS as ARRAY_BOTS, ScriptedBot, BotParams, make_bots

BOTS = {"random": random_bot.act, "econ": econ_bot.act}

"""Answer from the context: does the model answer a question with what a
short text just told it?

Every item is made here from a seed, so no item is in the corpus: a
template of TEMPLATES, filled with names, places, years, numbers and words
drawn from the pools of its language, states one fact, or two of the same
kind, in a short text; a question then asks for one of them in other
words, in one of the FORMATS of its language ("Question : … / Réponse :",
"Question: … / Answer:", "Utilisateur : … / Assistant :", "User: … /
Assistant:"). A document starts before the text (Tokenizer.END), as wpt
chat starts one. Measures, for each format:

- greedy and sampled: the share of items whose answer is in what the model
  writes after the prompt, up to its first line break, at most WRITTEN
  tokens, through its generation cache (wpt_inference.Writer, one row an
  item, stopped at the first token that holds a line break): greedy takes
  the likeliest token each time (top_k 1), sampled draws as wpt chat does
  (wpt_inference.TEMPERATURE and TOP_K, a generator seeded with SEED). The
  answer counts as a whole word, case aside, and when the text states two
  facts the line must not name the other one;
- rank and probability: the median rank (1: the likeliest) and the mean
  probability of the answer's first token, the answer read after a space
  right after the prompt.

The same figures are given for each kind of answer, whatever the format.
"""

import math
import random
import re
import statistics

from wpt_inference import TEMPERATURE, TOP_K
from wpt_tokenizer import Tokenizer
from wpt.eval import Result

SEED = 0
PROMPTS = 50  # of each format
WRITTEN = 12  # the most tokens written after a prompt

FORMATS = {
    "fr": {"question": "Question : {}\nRéponse :", "chat": "Utilisateur : {}\nAssistant :"},
    "en": {"question": "Question: {}\nAnswer:", "chat": "User: {}\nAssistant:"},
}
# (kind of answer, the text, the question, the slot the answer is in, the slot of the other fact or None)
TEMPLATES = {
    "fr": [
        ("city", "{name} habite à {city} depuis {small} ans. Chaque matin, {name} prend le bus pour aller au travail.",
         "Où habite {name} ?", "city", None),
        ("year", "Le pont a été construit en {year} par une entreprise de la région.",
         "En quelle année le pont a-t-il été construit ?", "year", None),
        ("job", "{name} travaille comme {job} dans un quartier tranquille.", "Quel est le métier de {name} ?",
         "job", None),
        ("pet", "{name} a un chien qui s'appelle {pet}. Ils se promènent tous les soirs.",
         "Comment s'appelle le chien de {name} ?", "pet", None),
        ("number", "La réunion aura lieu demain dans la salle {room}, au deuxième étage.",
         "Dans quelle salle aura lieu la réunion ?", "room", None),
        ("color", "La voiture de {name} est {color}. Elle est garée devant la maison.",
         "De quelle couleur est la voiture de {name} ?", "color", None),
        ("number", "Le train pour {city} part du quai {small}.", "De quel quai part le train pour {city} ?",
         "small", None),
        ("city", "{name} habite à {city} et {other_name} habite à {other_city}.", "Où habite {other_name} ?",
         "other_city", "city"),
        ("year", "La première école a ouvert en {year}, et la bibliothèque en {other_year}.",
         "En quelle année la bibliothèque a-t-elle ouvert ?", "other_year", "year"),
    ],
    "en": [
        ("city", "{name} has lived in {city} for {small} years. Every morning, {name} takes the bus to work.",
         "Where does {name} live?", "city", None),
        ("year", "The bridge was built in {year} by a local company.", "When was the bridge built?", "year", None),
        ("job", "{name} works as a {job} in a quiet part of town.", "What is {name}'s job?", "job", None),
        ("pet", "{name} has a dog called {pet}. They go for a walk every evening.",
         "What is the name of {name}'s dog?", "pet", None),
        ("number", "The meeting will take place tomorrow in room {room}, on the second floor.",
         "In which room will the meeting take place?", "room", None),
        ("color", "{name}'s car is {color}. It is parked in front of the house.", "What color is {name}'s car?",
         "color", None),
        ("number", "The train to {city} leaves from platform {small}.",
         "Which platform does the train to {city} leave from?", "small", None),
        ("city", "{name} lives in {city} and {other_name} lives in {other_city}.", "Where does {other_name} live?",
         "other_city", "city"),
        ("year", "The first school opened in {year}, and the library in {other_year}.",
         "When did the library open?", "other_year", "year"),
    ],
}
POOLS = {
    "fr": {
        "name": ["Marie", "Paul", "Julie", "Thomas", "Camille", "Lucas", "Sophie", "Hugo", "Léa", "Louis", "Emma",
                 "Nicolas", "Claire", "Antoine", "Chloé", "Pierre", "Sarah", "Julien", "Laura", "Mathieu"],
        "city": ["Lyon", "Nantes", "Lille", "Bordeaux", "Toulouse", "Marseille", "Strasbourg", "Rennes", "Nice",
                 "Grenoble", "Dijon", "Rouen", "Genève", "Bruxelles", "Montréal", "Tours"],
        "job": ["médecin", "architecte", "journaliste", "dentiste", "pilote", "photographe", "libraire", "juge",
                "peintre", "notaire", "pharmacien", "vétérinaire"],
        "pet": ["Max", "Rocky", "Filou", "Caramel", "Oscar", "Milo", "Biscuit", "Pirate", "Noisette", "Tigre",
                "Plume", "Volt"],
        "color": ["rouge", "bleue", "verte", "noire", "blanche", "jaune", "grise", "rose"],
    },
    "en": {
        "name": ["Mary", "John", "Emma", "James", "Sarah", "David", "Laura", "Peter", "Anna", "Thomas", "Emily",
                 "Daniel", "Kate", "Robert", "Lucy", "Michael", "Alice", "George", "Helen", "Mark"],
        "city": ["London", "Paris", "Berlin", "Madrid", "Rome", "Dublin", "Boston", "Chicago", "Sydney", "Toronto",
                 "Vienna", "Lisbon", "Denver", "Glasgow", "Seattle", "Oxford"],
        "job": ["doctor", "teacher", "baker", "nurse", "pilot", "lawyer", "farmer", "painter", "dentist",
                "journalist", "cook", "driver"],
        "pet": ["Max", "Buddy", "Rocky", "Charlie", "Oscar", "Milo", "Biscuit", "Daisy", "Pepper", "Shadow",
                "Rusty", "Bella"],
        "color": ["red", "blue", "green", "black", "white", "yellow", "grey", "pink"],
    },
}


def slots(language: str, draw) -> dict:
    """One value for every slot a template of language may name: two
    different ones for a kind that a text may state twice."""
    pools = POOLS[language]
    names, cities = draw.sample(pools["name"], 2), draw.sample(pools["city"], 2)
    years = draw.sample(range(1800, 2021), 2)
    return {"name": names[0], "other_name": names[1], "city": cities[0], "other_city": cities[1],
            "year": str(years[0]), "other_year": str(years[1]), "small": str(draw.randint(2, 30)),
            "room": str(draw.randint(100, 999)), "job": draw.choice(pools["job"]), "pet": draw.choice(pools["pet"]),
            "color": draw.choice(pools["color"])}


def items(draw) -> list[dict]:
    """PROMPTS items for each format of each language, each its own template and values."""
    found = []
    for language, formats in FORMATS.items():
        for name, form in formats.items():
            for _ in range(PROMPTS):
                kind, text, question, answer, other = draw.choice(TEMPLATES[language])
                values = slots(language, draw)
                found.append({"language": language, "format": f"{language}, {name}", "kind": kind,
                              "prompt": text.format(**values) + "\n\n" + form.format(question.format(**values)),
                              "answer": values[answer], "other": values[other] if other else None})
    return found


def named(line: str, value: str) -> bool:
    """Whether line holds value as a whole word, case aside."""
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", line, re.IGNORECASE) is not None


def answers(line: str, item: dict) -> bool:
    """Whether the line the model wrote gives the item's answer: it names
    the answer and not the other fact the text states."""
    return named(line, item["answer"]) and not (item["other"] and named(line, item["other"]))


def line_breaks(tokenizer) -> list[int]:
    """The tokens whose text holds a line break, and the end of a document:
    where a written answer stops."""
    return [Tokenizer.END] + [token for token in range(tokenizer.vocab_size)
                              if token != Tokenizer.END and b"\n" in tokenizer.decode([token])]


def first_line(tokenizer, tokens: list[int]) -> str:
    """The text of tokens up to the first line break or end of a document."""
    if Tokenizer.END in tokens:
        tokens = tokens[:tokens.index(Tokenizer.END)]
    return tokenizer.decode(tokens).decode("utf-8", "replace").split("\n")[0]


def written(context, prompts: list[list[int]], stops: list[int], generator: int, *, temperature, top_k) -> list[str]:
    """The first line the model writes after each prompt, through a writer
    of one row that stops at a line break or after WRITTEN tokens."""
    found = []
    for prompt in prompts:
        writer = context.model.writer(prompt, 1, temperature=temperature, top_k=top_k, stops=stops,
                                      generator=generator)
        found.append(first_line(context.tokenizer, writer.write({}, {0: WRITTEN})[0]))
    return found


def figures(found: list[dict]) -> dict:
    return {"prompts": len(found), "greedy": statistics.fmean(one["greedy"] for one in found),
            "sampled": statistics.fmean(one["sampled"] for one in found),
            "rank": statistics.median(one["rank"] for one in found),
            "probability": statistics.fmean(one["probability"] for one in found)}


def run(context) -> Result:
    tokenizer, model = context.tokenizer, context.model
    found = items(random.Random(SEED))
    prompts = [[Tokenizer.END] + tokenizer.encode(one["prompt"].encode()) for one in found]
    windows = [prompt + tokenizer.encode(f" {one['answer']}".encode()) for prompt, one in zip(prompts, found)]
    costs, ranks = model.losses(windows), model.ranks(windows)
    stops, generator = line_breaks(tokenizer), model.generator(SEED)
    likeliest = written(context, prompts, stops, generator, temperature=1.0, top_k=1)  # only the likeliest is drawn
    drawn = written(context, prompts, stops, generator, temperature=TEMPERATURE, top_k=TOP_K)
    for number, (prompt, one) in enumerate(zip(prompts, found)):
        # the loss and the rank at index len(prompt) - 1 are those of the answer's first token
        one.update(greedy_line=likeliest[number], sampled_line=drawn[number],
                   greedy=answers(likeliest[number], one), sampled=answers(drawn[number], one),
                   rank=int(ranks[number][len(prompt) - 1]),
                   probability=math.exp(-float(costs[number][len(prompt) - 1])))
    formats = {name: figures([one for one in found if one["format"] == name])
               for name in dict.fromkeys(one["format"] for one in found)}
    kinds = {kind: figures([one for one in found if one["kind"] == kind])
             for kind in dict.fromkeys(one["kind"] for one in found)}
    data = {"seed": SEED, "temperature": TEMPERATURE, "top_k": TOP_K, "written": WRITTEN, "formats": formats,
            "kinds": kinds, "items": found}
    lines = [f"A short text states a fact, or two of one kind, and a question asks for one in other words; "
             f"{PROMPTS} items made from templates for each format. greedy and sampled: the share of items whose "
             "answer the model writes on the line after the prompt, taking the likeliest token or sampling as wpt "
             f"chat does (temperature {TEMPERATURE}, the {TOP_K} likeliest tokens); rank and probability: the median "
             "rank (1: the likeliest) and the mean probability of the answer's first token right after the prompt.",
             "", "| | greedy | sampled | first token's rank | first token's probability |",
             "| --- | --- | --- | --- | --- |"]
    for name, one in list(formats.items()) + [(f"*{kind}*", one) for kind, one in kinds.items()]:
        lines.append(f"| {name} | {one['greedy']:.0%} | {one['sampled']:.0%} | {one['rank']:g} | "
                     f"{one['probability']:.3f} |")
    return Result(data, "\n".join(lines))

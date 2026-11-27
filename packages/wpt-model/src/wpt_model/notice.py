"""The license documents of a model folder.

Next to its weights, a model folder carries:

    LICENSE       the GNU AGPL, version 3: the license of the weights
    CREDITS       every text the model learned from, one line each, with its
                  credit and licenses, then the notices and license texts
                  they name (wpt.train.notice, written by wpt train from the
                  corpus)
    NOTICE        the licenses of the weights and of generated text, the
                  texts the model learned from (CREDITS), and the output
                  notice
    README.md     the model card: the same licenses first, then what the
                  model is, how it was built, and its evaluation results

write_documents(folder) generates NOTICE and README.md (documents(folder)
gives their text) from the folder's own files (config.json, provenance.json,
CREDITS, evals/*.json, drafter/config.json and drafter/provenance.json), so
they cannot disagree with what the folder holds. What they say of the
corpus comes from wpt_model.provenance, which reads a folder trained on a
snapshot ("Trained on snapshot S (change N, digest H) of the depot D,
order R, code at commit X"; the model card names the corpus's delivery
and the vocabulary's, for one trained on deliveries of the freight) and
one written before snapshots ("From corpus C, built from dataset T of the
depot D") alike. The drafter's weights are
the model's, under the same license and header; its prompts are the
beginnings of the plain texts the model read, so it reads no text the model
did not.
WEIGHT_METADATA goes into the header of every weight
file, so the licenses travel with the weights even alone.

OUTPUT-NOTICE and LICENSE in this package are copies of the repository's
root files; a test checks they stay identical.
"""

import json
from importlib.resources import files
from pathlib import Path

from wic_utils.programs import code_name

from .config import layer_kinds
from .provenance import corpus_id, corpus_label, deliveries, on_snapshot, order_name

LICENSE = "AGPL-3.0-or-later"
OUTPUT_LICENSE = "CC0-1.0"
REPOSITORY = "https://github.com/Workers-Intelligence-Commons/wpt"
NOTICE_NAME, CARD_NAME, LICENSE_NAME, EVALS_NAME, CREDITS_NAME = "NOTICE", "README.md", "LICENSE", "evals", "CREDITS"
DRAFTER_NAME = "drafter"  # the drafter's folder (drafter.py)
WEIGHT_METADATA = {"license": LICENSE, "output_license": OUTPUT_LICENSE,
                   "notice": f"NOTICE in the model folder; {REPOSITORY}"}
SUMMARY = (f"Weights: {LICENSE}. Generated text: {OUTPUT_LICENSE}, waived by the authors of WPT; "
           "material it reproduces from its training data or its context (prompt, documents, tool "
           "results) keeps its own license.")


def output_notice() -> str:
    return (files(__package__) / "OUTPUT-NOTICE").read_text(encoding="utf-8")


def license_text() -> str:
    return (files(__package__) / "LICENSE").read_text(encoding="utf-8")


def documents(folder) -> dict[str, str]:
    """{LICENSE, NOTICE and README.md: their text} for a model folder, from
    its config.json, provenance.json, CREDITS and evals/*.json."""
    folder = Path(folder)
    config = _read(folder / "config.json")
    provenance = _read(folder / "provenance.json")
    credits = (folder / CREDITS_NAME).read_text(encoding="utf-8")
    evals = [_read(path) for path in sorted((folder / EVALS_NAME).glob("*.json"))]
    drafter = folder / DRAFTER_NAME
    drafter = ({"config": _read(drafter / "config.json"), "provenance": _read(drafter / "provenance.json")}
               if (drafter / "config.json").exists() else None)
    return {LICENSE_NAME: license_text(), NOTICE_NAME: notice(provenance, credits),
            CARD_NAME: card(config, provenance, evals, drafter)}


def write_documents(folder):
    """Write documents(folder) into the folder. Replaces earlier copies."""
    for name, text in documents(folder).items():
        (Path(folder) / name).write_text(text, encoding="utf-8", newline="\n")


def notice(provenance, credits: str) -> str:
    """NOTICE: the licenses, what the model learned from
    (wpt_model.provenance.corpus_label), the texts of credits (CREDITS),
    then the output notice."""
    lines = [
        "WPT model notice", "",
        f"Weights: {LICENSE}. These weights are part of WPT (Workers' Pre-trained",
        "Transformer) and are licensed under the GNU Affero General Public License,",
        "version 3 or any later version; the license text is in LICENSE, next to",
        "this file.", "",
        f"Generated text: {OUTPUT_LICENSE}, under the output notice at the end of this file.", "",
        f"Project: {REPOSITORY}",
        *_built_from(provenance), "", "",
        "Texts this model learned from", "",
        *_learned_from(provenance),
        "one line per text, with its depot and id there, its title, authors and",
        "address when it has them, and its licenses (any one of them allows it),",
        "then the notices and the license texts the lines name.", "",
    ]
    return "\n".join(lines) + "\n" + credits.rstrip("\n") + "\n\n\n" + output_notice()


def card(config, provenance, evals, drafter=None) -> str:
    lines = [
        "---", "license: agpl-3.0", "---", "",
        f"# WPT model, corpus {corpus_id(provenance)[:12]}", "",
        f"> {SUMMARY}",
        ">",
        "> The full notice, with every text it learned from and its credit, is in",
        f"> [NOTICE](NOTICE). Project: {REPOSITORY}", "",
        "## Model", "",
        *_model_kind(provenance),
        f"- Parameters: {provenance['parameters']:,}",
        f"- Layers: {len(layer_kinds(config['window_layers'], config['sparse_groups'], config['sparse_layers']))} "
        f"({config['window_layers']} window layers on each side of an exact layer, then {config['sparse_groups']} "
        f"groups of an exact layer and {config['sparse_layers']} sparse layers), width {config['d_model']}; "
        f"experts: {config['n_experts']} routed "
        f"(top {config['top_k']}) and 2 shared per layer; Engram memory in layers "
        f"{', '.join(map(str, config['engram_layers'])) or 'none'}",
        f"- Context: {config['block_size']} tokens; vocabulary: {config['vocab_size']:,} tokens", "",
        "## Training", "",
        *_corpus_lines(provenance),
        *(f"- {line}" for line in _built_from(provenance)),
        f"- Device: {provenance['device']}",
        f"- Every input and tool version: [provenance.json](provenance.json)", "",
        "The steps, in order, each on the weights the one before it left:", "",
        *(f"{number}. {_step(step)}" for number, step in enumerate(provenance["steps"], start=1)), "",
        *_drafter(drafter, provenance),
        "## Texts", "",
        f"{provenance['texts']:,} texts; each one, with its authors and its credit: [NOTICE](NOTICE).", "",
        "## Evaluations", "",
    ]
    if not evals:
        lines += ["None yet: run `wpt eval <model folder>`.", ""]
    for result in evals:
        lines += [f"### {result['name']}", "", result["text"].rstrip(), "",
                  f"`{result['file']}` (sha256 `{result['file_sha256'][:12]}`), {_eval_code(result)}, "
                  f"on {result['device']}. Data: [evals/{result['name']}.json](evals/{result['name']}.json)", ""]
    return "\n".join(lines)


def _eval_code(result) -> str:
    """The code a result was measured with: code() of wic_utils.programs
    in a result that names its code, wpt eval's package version in an
    older one that names it."""
    return f"code {code_name(result['code'])}" if "code" in result else f"wpt eval {result['wpt_eval']}"


def _drafter(drafter, provenance):
    if drafter is None:
        if (provenance.get("options") or {}).get("drafter") is False:
            return ["## Drafter", "",
                    "None: `wpt train --no-drafter` saved the model without one, so it writes alone, without "
                    "speculative decoding. The same `wpt train` command without that option trains one.", ""]
        return []
    config, provenance = drafter["config"], drafter["provenance"]
    lines = ["## Drafter", "",
             f"A DSpark drafter for speculative decoding (`drafter/`): {config['layers']} layers that read the "
             f"hidden states of layers {', '.join(map(str, config['features']))}, up to {config['block']} drafts per "
             "round. The model checks every draft and keeps it only by the rule that leaves its own distribution "
             "unchanged: the drafter changes how fast the model writes, never what it writes.", ""]
    trained, target = provenance.get("trained_by"), (provenance.get("target") or {}).get("weights")
    if trained:
        corpus = provenance.get("corpus")
        lines.append(f"- Trained by `{trained['command']}`, order `{order_name(trained)}`, the model frozen, on "
                     "answers the model wrote itself to the beginnings of the plain texts of corpus "
                     f"`{(corpus or 'unknown')[:12]}`")
    if target:
        lines.append("- Against the weights " + ", ".join(f"`{name}` (sha256 `{sha[:12]}`)"
                                                           for name, sha in target.items()))
    measure = provenance.get("acceptance")
    if measure:
        lines.append(f"- On held-out prompts: {measure['accepted']} of {measure['verified']} drafts kept, "
                     f"{measure['tokens_per_round']:.2f} tokens per pass of the model")
    return lines + [""]


TEXT = "windows"  # what a step that reads text reads, as provenance.json's steps say (`reads`)


def _model_kind(provenance):
    if all(step["reads"] == TEXT for step in provenance["steps"]):
        return ["A WPT mixture-of-experts model (window, exact and sparse attention,",
                "Attention Residuals, Engram memory). It is trained on text only: it continues",
                "text and follows no instructions.", ""]
    return ["A WPT mixture-of-experts model (window, exact and sparse attention,",
            "Attention Residuals, Engram memory), trained by the steps below.", ""]


def _step(step):
    """One line of the card for a step of provenance.json's `steps`."""
    stages = step["stages"]
    limit = sum(stage["steps"] for stage in stages)
    text = (f"`{step['step']['name']}`: method {step['method']['name']}, optimizer {step['optimizer']['name']}, "
            f"environment {step['environment']['name']}"
            + (f", locked {', '.join(step['locked'])}" if step["locked"] else "")
            + f"; {step['updates']:,} updates (limit {limit:,}), the weights of update {step['weights_update']:,}")
    if any(stage["block_size"] is not None for stage in stages):
        text += "; " + ", ".join(f"{stage['block_size']:,}-token windows, {stage['batch_size']:,} an update, for "
                                 f"{stage['steps']:,} updates" for stage in stages)
    else:
        text += f"; {stages[0]['batch_size']:,} examples an update"
    asked, read = step.get("settings", {}).get("readings", 1), step.get("learned", {}).get("plain")
    if read and any(stage["block_size"] is not None for stage in stages):
        times = sum(stage["steps"] * stage["batch_size"] * stage["block_size"] for stage in stages) / read
        text += f"; its {read:,} tokens of plain text read {times:.2f} times (readings {asked})"
    if step["best_validation_loss"] is not None:
        text += f"; best validation loss {step['best_validation_loss']:.4f}"
    if step.get("validation") is not None:
        text += ("; validated on held-out text" if step["validation"] == "held_out"
                 else "; validated on training text, as its corpus has no held-out portion")
    if step["reused"]:
        text += f"; the result of an earlier run (`{step['result'][:12]}`)"
    return text + "."


def _learned_from(provenance):
    """The NOTICE's line that heads the texts: what the model was trained on."""
    if on_snapshot(provenance):
        return [f"Trained on {corpus_label(provenance)}, corpus {corpus_id(provenance)}:"]
    return [f"From {corpus_label(provenance)}:"]


def _corpus_lines(provenance):
    """The model card's lines on the corpus."""
    delivered = deliveries(provenance)
    if delivered:
        size = ((provenance["corpus"].get("request") or {}).get("selection") or {}).get("size") or {}
        built, wanted = size.get("amount"), (provenance.get("data") or {}).get("total_tokens")
        again = "the same request to the freight makes it again"
        if built != wanted:
            again = (f"made at {_tokens(built)} tokens, not the order's {_tokens(wanted)}: the same request to the "
                     "freight makes it again, `corpus.request` in provenance.json")
        vocabulary = delivered["vocabulary"]
        return [f"- Trained on {corpus_label(provenance)}",
                f"- Corpus: `{corpus_id(provenance)}`, delivery `{delivered['corpus']['key']}` (a cache: {again})",
                f"- Vocabulary: delivery `{vocabulary['key']}`"
                + (" (adopted from the depot's tokenizer recipe "
                   f"{vocabulary['adopted'].get('recipe')} version {vocabulary['adopted'].get('version')})"
                   if "adopted" in vocabulary else f" (made by {vocabulary['preparation']['name']})")]
    if on_snapshot(provenance):
        built = (provenance["corpus"].get("data") or {}).get("total_tokens")
        wanted = (provenance.get("data") or {}).get("total_tokens")
        again = "the same snapshot, order and code build it again"
        if built is not None and built != wanted:
            again = (f"built at {built:,} tokens, not the order's {_tokens(wanted)}: the same snapshot and code "
                     "build it again from its own data section, `corpus.data` in provenance.json")
        return [f"- Trained on {corpus_label(provenance)}", f"- Corpus: `{corpus_id(provenance)}` (a cache: {again})"]
    corpus = provenance["corpus"]
    return [f"- Corpus: `{corpus['id']}`, built from dataset `{corpus['dataset']}` of the depot `{corpus['depot']}`"]


def _tokens(count) -> str:
    return "no size" if count is None else f"{count:,}"


def _built_from(provenance):
    commit = provenance.get("repository_commit")
    if commit is None:
        return ["Built outside a git checkout: the repository commit is unknown."]
    dirty = " with uncommitted changes" if provenance.get("repository_dirty") else ""
    return [f"Built from repository commit {commit}{dirty}."]


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))

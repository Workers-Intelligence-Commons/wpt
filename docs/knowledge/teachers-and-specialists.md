# Does a small model learn more from other models than from the text alone?

This file asks one question in two forms: whether a small model gains from
imitating a larger model trained on the same corpus, and whether several
small models, each trained on one domain, can be brought together into one
general model of the same size. No model from outside may be used: every
teacher is trained from scratch on WPT's corpus. Its marks are defined in
[README.md](README.md).

## Answer

The question is open: WPT has no code for either form.

- On a very small corpus, 10M words, a student that imitates teachers
  trained on the same data beats the same model trained alone on most
  tests. **published, agreed**
- With billions of tokens, and the teacher's training counted in the cost,
  imitation does not pay: it wins only under a threshold of tokens and when
  the teacher already exists. **published, one source**
- A larger teacher is not always a better one. **published, agreed**
- Imitation can harm what the student does with its context, copying
  included. **published, one source**
- Averaging the weights of models that never shared a start fails: the
  result is no model at all. A shared first phase, or frequent averaging
  during training, repairs it. **published, agreed**
- An average of domain specialists that shared the first half of training
  beats one model of equal compute on the domains seen, and loses on unseen
  domains at 125M parameters. **published, one source**
- The working assumption for one GPU: specialists trained at the same time
  share the GPU, so they bring no time; the gain can only be in quality.
  The form worth a measure is a common model trained first, then branches
  by domain, then one average. A series of sizes already trains a larger
  model beside each small one, so a teacher costs nothing more there.
  **hypothesis**

## Evidence

All figures from published sources in this section are **unchecked**: they
were read through an automatic reader of the papers' pages. Each source
exists under the id or address given.

### A student and its teachers

- *Baby Llama* (2023, arXiv 2308.02019). A student of 58M and teachers of
  705M and 360M, all trained on the same 10M words. The student beats the
  same model trained alone on most grammar tests and loses on some. Two
  teachers beat 4 or 6.
- *BabyLlama-2* (2024, arXiv 2409.17312). A student of 345M with two
  teachers of its size, 9.5M words: 64.0 against 61.3 to 63.6 for the
  teachers. The authors do not know where the gain comes from.
- *Born Again Neural Networks* (2018, arXiv 1805.04770). The same model
  taught by itself: perplexity 71.87 to 68.56 at 52M, only when the loss on
  the text is kept beside the teacher's.
- *Distillation Scaling Laws* (2025, arXiv 2502.08606). 143M to 12.6B. With
  the teacher's training counted, the best student is always worse than a
  model trained on the text; a stronger teacher can make a worse student.
- *Pre-training Distillation: A Design Space Exploration* (2024, arXiv
  2410.16215). Teachers of 9B and 32B. A student of 330M loses, 32.1
  against 32.6; one of 6.8B gains, 49.8 against 44.9. It pays when the
  student has a tenth of the teacher's size or more.
- *Distilled Pretraining* (2025, arXiv 2509.01649). Teacher of 8B, student
  of 1B, the same data. Imitation harms learning from the context and the
  heads that copy; leaving it out on the tokens the teacher is surest of
  repairs part of it.
- arXiv 2410.18779 (2024). Imitating for the whole run hurts, 2.071 against
  1.868; for the first sixth of the run it gives 1.863 and 28% less time.
- *Large scale distributed neural network training through online
  distillation* (2018, arXiv 1804.03235). Two models that teach each other
  reach the same error in half the updates on separate data, and gain
  little on the same data.
- Against: arXiv 2411.16487 (2024). On 10M words a plain teacher gives
  47.65 against 49.62 for the model alone.

### Specialists brought together

- *Branch-Train-Merge* (2022, arXiv 2208.03306). 8 domains, specialists of
  125M to 1.3B after a shared phase of half the updates. Without a shared
  phase the average has a perplexity in the thousands. With it, against one
  model of equal compute: 18.0 against 19.9 on seen domains and 28.5
  against 25.2 on unseen ones at 125M; 13.4 against 14.2 and 17.0 against
  18.4 at 1.3B. The best shared phase is 60% to 70% of training.
- *Linear Mode Connectivity and the Lottery Ticket Hypothesis* (2019, arXiv
  1912.05671). Two copies can be averaged after a shared start of 1.5% to
  20% of training, by task. Images only.
- *PAPA* (2023, arXiv 2304.03094). Models averaged once at the end: 9.26%
  against 79.35% for each. Pulled toward their average every 10 updates:
  82.1 against 80.2. Images only.
- *DiLoCo* (2023, arXiv 2311.08105). 60M to 400M, 8 workers on data skewed
  by domain, averaged every 500 updates: perplexity 15.02, against 14.72
  for one model given as many updates in all.
- *Git Re-Basin* (2022, arXiv 2209.04836) and arXiv 2403.00986 (2024).
  Matching the units of two models before averaging works for wide image
  models and leaves a barrier for text models.
- *Branch-Train-MiX* (2024, arXiv 2403.07816). Specialists become the
  experts of one larger model; the result is larger than a specialist.
- *Distilling the Knowledge in a Neural Network* (2015, arXiv 1503.02531).
  61 specialists; putting them back into one model is left undone.

## Open

- **A teacher counted in the cost, on 1B to 5B tokens, under 500M
  parameters.** Published work has 10M words on one side and hundreds of
  billions of tokens on the other. A measure trains a small model alone and
  the same model with the next size of its series as teacher, and compares
  the loss, the copying test of `wpt eval` and the time.
- Whether imitation delays or brings forward copying at these sizes.
- **Specialists of one small size against one model.** A measure trains a
  common model for half the updates, branches by tag of the snapshot,
  averages, and compares with one model of equal compute on seen and unseen
  domains. No source does it under 125M.
- Putting specialists trained from scratch back into one model of the same
  size by imitation, against one model of equal compute: not measured.
- Whether WPT's routed experts already do, inside one model, what domain
  specialists would do.

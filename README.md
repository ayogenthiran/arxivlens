# ArxivLens

ArxivLens is a paper-discovery tool for Master's and PhD students. A student describes a research question. The app returns distinct arXiv papers and a short summary whose claims point back at those papers.

## The problem

arXiv is too large to read paper by paper. A keyword search on titles misses work that answers the question in the abstract. A chat model asked the same question will write an answer that sounds specific and still cite the wrong paper, or quote a sentence that is not in the paper.

Two earlier retrieval choices made that worse. Cutting each paper into overlapping passages let one paper fill the whole result list. Searching the full archive meant keyword search could not stay in memory, so ranking used vectors alone and mixed weak neighbors into the list. Rewriting the student's question before search also changed what they asked, and a stronger paper was replaced by a weaker one.

## How we solved it

The live set is 5,000 recent papers in computation and language and machine learning (`cs.CL` and `cs.LG`), from 2024 onward. Each paper is stored once: the title, then the full abstract. Search covers that metadata.

A question is embedded with the same MiniLM model used when the papers were indexed. On this corpus, vector rank and keyword rank are fused, so a paper has to match both the meaning of the question and its words. A cross-encoder then rereads the short list and drops papers that are only barely related. Category, author, and year filters run before that final cut, so the list stays full of papers that match the filter.

Those papers appear as soon as search returns. The model writes the summary only from the papers on the page. A citation is kept only when its quote appears in that paper, and each citation opens the abstract on arXiv. If the summary fails, the papers stay. The question is searched as the student typed it. Rewriting it before search stayed off, because the original wording retrieved the stronger papers.

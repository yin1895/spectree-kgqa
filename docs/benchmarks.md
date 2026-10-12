# Dataset inventory

- **WebQSP**: entity-linked Freebase QA. Convert questions with `--format webqsp`, retain official question IDs.
- **CWQ 1.1**: compositional Freebase QA. Convert with `--format cwq` and preserve source-family identifiers for resampling when supplied.
- **GrailQA**: Freebase QA with IID, compositional and zero-shot schema-generalization levels. Use a fixed, stratified public-development population, and keep tuning separate.
- **MetaQA-3hop**: movie KG QA with a public WikiMovies graph; entity links and relation normalization follow official data.
- **PathQuestion-Large**: optional directed-relation path-depth check.

Each benchmark has its own official normalization/scorer convention. The canonical format records the question/root/answer-ID sets, so the same SpecTree inference pipeline can be used once source-specific conversion is complete.

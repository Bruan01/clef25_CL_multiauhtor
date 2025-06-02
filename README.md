# About
this project is atemplate that uesd to submit the software with docker 

upload submission to TIRA (remove `--dry-run`):

```
tira-cli code-submission \
	--mount-hf-model microsoft/deberta-base \
	--path . \
	--task multi-author-writing-style-analysis-2025 \
	--dataset multi-author-writing-spot-check-20250503-training \
	--dry-run
```


# TODO: Headline Generator Improvements ✅ COMPLETED

## Issue 1: Fine-tuned model produces same topic with just 'a'/'the' variations ✅
- [x] `apps/local_model_client.py`: Switched from deterministic beam search to diverse sampling (`do_sample=True`, temperature=0.85, top_k=50, top_p=0.92)
- [x] `apps/local_model_client.py`: Added `seed` parameter for regeneration

## Issue 2: Groq headline length should follow max_chars like fine-tuned model ✅
- [x] `apps/groq_client.py`: Updated prompt with strict length directives (80%–100% of max_chars) instead of weak "aim for approximately" phrasing
- [x] `apps/groq_client.py`: Added `seed` parameter for regeneration
- [x] `app.py`: Removed separate "Groq headline length" slider — both backends now use the same `max_chars` slider

## Issue 3: Regenerate new headlines on re-click ✅
- [x] `apps/pipeline.py`: Added `seed` parameter to pipeline functions, passes through to model clients
- [x] `app.py`: Tracks `generation_id` in session state, increments on each click, passes to pipeline

<!-- ## Bonus Fix: Display options checkbox behavior ✅
- [x] `app.py`: Warnings expander and raw output expander now only show when their respective checkboxes are ticked -->

## Documentation ✅
- [x] `README.md`: Fully rewritten with project overview, tech stack, project structure, setup instructions, architecture, features, and known limitations

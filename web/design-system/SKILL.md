---
name: citeclaw-design
description: Use this skill to generate well-branded interfaces and assets for CiteClaw, either for production or throwaway prototypes/mocks. Contains the design guidelines, color + type tokens, fonts, the brand mark, reusable components, and full-screen UI kits for the CiteClaw WebUI (Build / Run / Explore).
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files (tokens/, components/, ui_kits/, guidelines/, assets/).

CiteClaw is a literature-acquisition tool; its WebUI is a warm-cream, plum-ink, forest-green-accented workspace with three modes (Build / Run / Explore). The single most important rule: **green (#3a7550) is reserved for critical moments only** — primary action, active tab, selection, live/running state, seed nodes, brand, links. Everything else is cream + ink.

If creating visual artifacts (slides, mocks, throwaway prototypes), copy assets out and create static HTML files that link `styles.css`. If working on production code, copy assets and apply the tokens + component contracts directly.

If the user invokes this skill without other guidance, ask what they want to build, ask a few questions, and act as an expert CiteClaw designer who outputs HTML artifacts or production code as appropriate.

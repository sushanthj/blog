---
name: write-knowledge-base-post
description: Skill for writing didactic knowledge-base blog posts in Sush's style. Trigger when creating or editing posts in _posts/knowledge-base/.
---

# Writing Knowledge Base Posts

## Voice & Tone

- **Conversational, not academic.** Write like you're explaining to a friend at a whiteboard. Incomplete sentences are fine.
- **No meta-descriptions.** Never write "This post walks through..." or "We'll cover..." — just start teaching.
- **First person plural is okay** ("we take robot state and output...") but don't overdo it.

## Structure

- **Ground the reader first.** The very first section should establish the simplest possible mental model before introducing any complexity. Name the section after the core intuition, not the academic term. Example: `# *Neural Nets are the 'Policy'*` not `# Introduction to Policy Networks`.
- **Italics in H1s for emphasis** on the key intuition phrase: `# *Neural Nets are the 'Policy'*`
- **Visuals before text.** Prefer placing a GIF/animation/diagram *before* the paragraph that explains it. If a visual doesn't exist yet, leave a `TODO:` describing what it should show so it can be created later.
- **Titles are practical, not jargon.** "Using Neural Nets for RL" not "Imitation Learning". Describe what you're *doing*, not the academic field name.
- **Subtitles are short.** One line, no fluff.
- **Permalinks suggest series.** Use naming like `/basics_1/`, `/basics_2/` when posts are part of a progression.

## Content

- **Build up from the simplest case.** Start with the dead-simple version (e.g. "a box that takes state and outputs action"), then layer on why it's insufficient, then the next approach, etc.
- **Bold the key takeaway** inline in the paragraph where it appears. Don't create separate callout sections.
- **Use TODO markers** for planned visuals or sections that need more work: `TODO: Insert a GIF of...` with a description of what the visual should convey.
- **Keep paragraphs short.** 2-3 sentences max before a break, visual, or new heading.

## Animations & Visuals

- **Always use the 3b1b animation creator** at `/home/sush/repos/3b1b_animation_creator/` for generating GIFs and animations. Read its `CLAUDE.md` for the full workflow (Docker-based Manim rendering, output folder structure, quality flags).
- Render blog GIFs at medium quality 720p30fps and since Manim supports output format as GIF directly, use that to avoid unnecessary conversions. If you need to do post-processing (like cropping or resizing), do it in the animation creator workflow before outputting the final GIF.
- If anything needs to be 'sudo apt install'-ed for the animation workflow do it in the `3b1b_animation_creator` docker container. It should already have everything needed for Manim and GIF processing (like gifsicle, ImageMagick, and Python libraries). If you add new dependencies, update the Dockerfile and document them in `CLAUDE.md`.
- Place output GIFs in `/home/sush/repos/blog/images/knowledge_base/concepts/<topic>/` matching the post's image directory structure.
- ENSURE GIFs have no overlapping text or visuals that could be hard to read when rendered at 720p. Use Manim's text formatting and positioning features to keep everything clear and legible!!!!
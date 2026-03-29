---
layout: projects-post
title: 3B1B Animation Skills
subtitle: AI-powered Manim animations from natural language
featured_image: /images/projects/3b1b_animations/cover.gif
categories: projects
---

![](/images/projects/3b1b_animations/cover.gif)

Inspired by Grant Sanderson's [3Blue1Brown](https://www.3blue1brown.com/) — I've been following his animations since 2017 and I would not know half the math concepts without his animations. 

I've always wanted to learn to do Manin animations but couldn't make time for it. Now with the help of AI agents this is totally possible! 

[GitHub](https://github.com/sushanthj/3B1B_animation_skills)

## How It Works

This project is just a collection of skills that let AI coding assistants generate those same style of Manim animations from natural language prompts.

Clone the repo, run `./launch.sh`, and open it in your AI coding tool (Claude Code, Cursor, Windsurf, Copilot). Describe what you want to animate — text, `.pptx` slides, images, or research papers — and the video renders automatically.

![](/images/projects/3b1b_animations/typing_prompt.gif)

![](/images/projects/3b1b_animations/robot_arm.gif)

## Tech Stack

Manim CE + Docker + LaTeX + ffmpeg, with a collection of skills encoding animation best practices and crash-prevention wrappers for common Manim pitfalls.

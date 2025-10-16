---
layout: default
title: Projects
description: Project Portfolio
parent: Continuing Learning
nav_order: 3
has_children: false
permalink: /knowledge-base/projects/
---

<section class="intro">
    <div class="wrap">
        <h1>Projects</h1>
        <p></p>
    </div>
</section>

<section class="portfolio">
    <div class="content-wrap portfolio-wrap">
        {% for post in site.posts reversed %}
        {% if post.categories contains 'projects' %}
        <div class="portfolio-item">
            <a class="portfolio-item__link" href="{{ post.url | relative_url }}">
                <div class="portfolio-item__image">
                    <img src="{{ post.featured_image | relative_url }}" alt="{{ post.title }}">
                </div>

                <div class="portfolio-item__content">
                    <div class="portfolio-item__info">
                        <h2 class="portfolio-item__title">{{ post.title }}</h2>
                        <p class="portfolio-item__subtitle">{{ post.subtitle }}</p>
                    </div>
                </div>
            </a>
        </div>
        {% endif %}
        {% endfor %}
    </div>
</section> 
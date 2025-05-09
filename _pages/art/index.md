---
layout: default
title: Art
description: Engineering Art
featured_image: /images/social.jpg
permalink: /art/
---

<section class="intro">
	<div class="wrap">
		<h1>Art with some Tech</h1>
		<p>I love 3D printing and using digital art tools. Here's a few cool things I've made!</p>
	</div>
</section>

<section class="portfolio">
	<div class="content-wrap portfolio-wrap">
		{% for post in site.posts reversed %}
		{% if post.categories contains 'art' %}
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
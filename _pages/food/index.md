---
layout: default
title: Food
description: Culinary adventures and recipes
featured_image: /images/social.jpg
permalink: /food/
---

<section class="intro">
	<div class="wrap">
		<h1>Culinary Shenanigans</h1>
		<p>Recipes fine-tuned via extensive ablations on the random contents of my kitchen cabinets.</p>
	</div>
</section>

<section class="portfolio">
	<div class="content-wrap portfolio-wrap">
		{% for post in site.posts reversed %}
		{% if post.categories contains 'food' %}
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
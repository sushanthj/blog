---
layout: default
title: Travel
description: Travel adventures and destinations
featured_image: /images/social.jpg
---

<section class="intro">
	<div class="wrap">
		<h1>Travel</h1>
		<p>Travel adventures and destinations</p>
	</div>
</section>

<section class="portfolio">
	<div class="content-wrap portfolio-wrap">
		{% for post in site.posts reversed %}
		{% if post.categories contains 'travel' %}
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
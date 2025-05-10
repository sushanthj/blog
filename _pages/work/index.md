---
layout: default
title: Work
description: My work experience and education
permalink: /work/
---

<section class="intro">
	<div class="wrap">
		<h1>Experience</h1>
	</div>
</section>

<section class="single">
	<div class="wrap" style="max-width: 90%;">
		<figure>
			<img src="./images/MRSD.png" alt="MRSD Degree" class="full-width">
			<figcaption class="image-caption">Master's in Robotic Systems Development</figcaption>
		</figure>

		<p>In 2024, I earned my Master's in Robotic Systems Development (MRSD) from Carnegie Mellon University, specializing in Computer Vision.</p>

		<hr>

		<h2>Current Work</h2>

		<p>I currently work at Intuitive Surgical, where I focus on:</p>
		<ul>
			<li>Computer vision algorithms for surgical robotics</li>
			<li>Real-time image processing</li>
			<li>Machine learning for medical applications</li>
		</ul>

		<hr>

		<h2>Previous Work</h2>

		<p>Part-time research at <a href="https://labs.ri.cmu.edu/kantorlab/research/">Kantor Lab</a> at CMU's Field Robotics Center, working on Robotics for Agriculture. I work on a robotic grapevine pruning project involving 3D skeletonization of grapevine pointclouds for pruning weight estimation.</p>

		<figure>
			<img src="./images/pure_vine.gif" alt="Grapevine Pruning" class="full-width">
			<figcaption class="image-caption">Robotic Grapevine Pruning Project</figcaption>
		</figure>

		<hr>

		<figure>
			<img src="./images/magna_homepage.png" alt="Magna" class="full-width">
		</figure>

		<hr>

		<figure>
			<img src="./images/ts_homepage.png" alt="Niqo Robotics" class="full-width">
		</figure>

		<p>I worked on the perception team at <a href="https://www.youtube.com/watch?v=Sap3Z-9vSow">Niqo Robotics</a>.</p>

		<hr>

		<figure>
			<img src="./images/edhitha_homepage.png" alt="Edhitha" class="full-width">
		</figure>

		<p>Edhitha was a student team which has continuously taken part at the <a href="https://suas-competition.org/">AUVSI SUAS</a> competition held at Maryland, USA. I was part of the 2016 team where we finished 5th amongst 60 international teams and was the team lead in 2017.</p>

		<div class="video-container">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/oVnpnDw6jZ0?controls=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
		</div>

		<hr>

		<h2>Resume</h2>
		<div class="resume-container">
			<iframe 
				src="https://drive.google.com/file/d/1AnelKYN5vzHbun0cMJ41QDXI-L0FsC3t/preview?usp=sharing" 
				width="100%" 
				height="800px" 
				allow="autoplay">
			</iframe>
			<a href="https://drive.google.com/uc?export=download&id=1AnelKYN5vzHbun0cMJ41QDXI-L0FsC3t" class="download-btn" download>
				<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<path d="M12 16L12 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
					<path d="M9 13L12 16L15 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
					<path d="M20 16.7428C21.2215 15.734 22 14.2079 22 12.5C22 9.46243 19.5376 7 16.5 7C16.2815 7 16.0771 6.886 15.9661 6.69774C14.6621 4.48484 12.2544 3 9.5 3C5.35786 3 2 6.35786 2 10.5C2 12.5661 2.83545 14.4371 4.18695 15.7935" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
					<path d="M8 16V19C8 20.1046 8.89543 21 10 21H14C15.1046 21 16 20.1046 16 19V16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
				Download Resume
			</a>
		</div>
	</div>
</section>

<style>
.resume-container {
	margin: 20px auto;
	position: relative;
	max-width: 800px;
}

.resume-container iframe {
	width: 100%;
	height: 800px;
	border: none;
	margin: 0;
	padding: 0;
	display: block;
}

.download-btn {
	position: absolute;
	bottom: 12px;
	right: 12px;
	display: inline-flex;
	align-items: center;
	gap: 8px;
	padding: 8px 16px;
	background-color: #f5f5f5;
	border-radius: 4px;
	color: inherit;
	text-decoration: none;
	font-size: 14px;
	transition: background-color 0.2s ease;
	box-shadow: 0 2px 4px rgba(0,0,0,0.1);
	z-index: 1;
}

.download-btn:hover {
	background-color: #e5e5e5;
}

.download-btn svg {
	width: 20px;
	height: 20px;
}

.video-container {
	max-width: 800px;
	margin: 20px auto;
}

@media screen and (max-width: 768px) {
	.resume-container iframe {
		height: 600px;
	}
	
	.resume-container,
	.video-container {
		max-width: 100%;
	}
}
</style> 
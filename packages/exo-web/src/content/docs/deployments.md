---
title: Deployments
description: Deploy agents as APIs and widgets
section: guides
order: 6
---

# Deployments

Deploy your agents as API endpoints, embeddable chat widgets, or internal tools.

## Deployment Types

### API Endpoint

Expose your agent as a REST API:

- **POST** `/api/v1/deployments/:id/chat` — Send a message and get a response
- Supports streaming via Server-Sent Events
- Authenticate with API keys

### Chat Widget

Embed a chat interface on any website:

1. Go to **Deployments** and select your deployment
2. Click **Widget** to get the embed code
3. Paste the `<script>` tag into your HTML

### Internal Tool

Share agents with your team through the Exo dashboard — no external deployment needed.

## Configuration

- **Rate limiting** — Set requests per minute per client
- **Allowed origins** — CORS configuration for widget embeds
- **Authentication** — API key or session-based auth
- **Versioning** — Pin to a specific agent version or use latest

# sesha-2k3.github.io

My personal blog: notes on machine learning and LLM systems, write-ups of papers I read, tutorials, and things I build.

**Live site:** [sesha-2k3.github.io](https://sesha-2k3.github.io)

## What's here

- **Posts:** long-form writing and study notes, including the **RAG from Scratch** series, which builds a retrieval-augmented generation pipeline piece by piece. The code for the series lives in [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality).
- **Reading log:** articles, papers, and books I've read, with a short note on why each one stuck.
- **About:** who I am and what I've built.

## Tech stack

- [Astro](https://astro.build/), built on the [AstroPaper](https://github.com/satnaing/astro-paper) theme
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [Pagefind](https://pagefind.app/) for static search
- [GitHub Pages](https://pages.github.com/) for hosting, deployed with GitHub Actions

## Project structure

```
/
├── .github/workflows/deploy.yml   # builds and deploys to GitHub Pages on push to main
├── public/                        # static files: favicon, default share image
├── src/
│   ├── content/
│   │   ├── pages/about.md         # About page text
│   │   ├── posts/                 # blog posts (Markdown / MDX)
│   │   └── reading/               # reading log entries
│   ├── pages/
│   │   └── reading.astro          # reading log page
│   ├── components/                # header, footer, and other UI pieces
│   ├── styles/                    # global styles and color schemes
│   └── content.config.ts          # content collection schemas
├── astro-paper.config.ts          # site settings: title, socials, share links, features
└── astro.config.ts
```

## Running locally

Requires Node.js 22.22 or newer.

```bash
npm install
npm run dev       # local dev server at http://localhost:4321
npm run build     # type-check, build, and index for search
npm run preview   # preview the production build
```

## Writing a post

Add a Markdown file to `src/content/posts/`. The file name becomes the URL, so `my-post.md` is published at `/posts/my-post/`.

```markdown
---
title: "Post title"
pubDatetime: 2026-10-21T00:00:00Z
description: "One-line summary for the post list and link previews."
tags:
  - rag
featured: false
draft: true
---

Post content in Markdown.
```

Posts with `draft: true` appear locally but are not published. Set it to `false` when the post is ready, then push to `main`.

## Adding to the reading log

Add a Markdown file to `src/content/reading/`:

```markdown
---
title: "Article title"
author: "Author or publication"
type: article        # article, blog, paper, tweet, thread, newsletter, video, podcast, or book
url: "https://..."
date: 2026-10-21
---

A short note on why it stuck.
```

For a book in progress, use `type: book` and `status: reading`. It appears under "Currently reading" until you change it to `status: finished`.

## Deployment

Every push to `main` runs `.github/workflows/deploy.yml`, which builds the site with Node 24 and publishes it to GitHub Pages. Check the **Actions** tab if a deploy fails.

## Credits

Built on [AstroPaper](https://github.com/satnaing/astro-paper) by [Sat Naing](https://satnaing.dev) and contributors, licensed under the MIT License. The theme's original license is kept in [LICENSE](LICENSE).

Blog posts and other written content are my own. Please ask before republishing them.

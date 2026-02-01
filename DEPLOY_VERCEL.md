# Deploy to Vercel (Astro)

## Prereqs
- A GitHub repository that contains this project
- A Vercel account connected to GitHub

## 1) Set the correct SITE.website
Edit `src/config.ts`:
- `SITE.website` must be your final domain, e.g.
  - `https://<project>.vercel.app`
  - or your custom domain

This affects sitemap/RSS canonical URLs.

## 2) Push to GitHub
From `/mnt/data/tech-blog`:
```bash
git status
git add -A
git commit -m "init tech blog mvp"
# then add your remote and push
# git remote add origin <GITHUB_REPO_URL>
# git push -u origin main
```

## 3) Import into Vercel
- Vercel Dashboard → **Add New… → Project**
- Import the GitHub repo
- Framework preset should auto-detect **Astro**
- Build command: `npm run build`
- Output directory: `dist`

We also have `vercel.json` in the repo (should be auto-respected).

## 4) Deploy
- Click **Deploy**
- After deployment, update `SITE.website` to the deployed URL and redeploy.

## 5) Optional: Custom domain
- Project → Settings → Domains → add your domain


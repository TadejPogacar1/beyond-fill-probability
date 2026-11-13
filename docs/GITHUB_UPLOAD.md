# Publish to GitHub

The local repository is initialized on the `main` branch and all project files
are staged. Configure your identity, create the first commit, and publish:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
git commit -m "Initial public release"
gh repo create queue-aware-market-making-holdout \
  --public \
  --source=. \
  --remote=origin \
  --push
```

If GitHub CLI is unavailable, create an empty repository in the GitHub web
interface and follow its commands for pushing an existing repository. Do not add
a remote README, license, or `.gitignore`; they already exist locally.

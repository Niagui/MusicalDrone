# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.

## Code of Conduct
We follow the Contributor Covenant Code of Conduct.
All participants are expected to:
Be respectful and collaborative in discussions and reviews.


Avoid discriminatory or harassing behavior.

## Getting Started

Prerequisites
Python ≥ 3.10
Requires WSL
Setup
```git clone https://github.com/Niagui/MusicalDrone.git
cd musical drone
pip install -r requirements.txt
```
Run the individual .py file

## Branching & Workflow

We use trunk-based development with feature branches off main.

main → stable, production-ready

feature/<short-description> → individual work
Commit → PR → Review → Merge.
Rebase locally before merging to maintain a linear history:
```git fetch origin```
```git rebase origin/main```


## Issues & Planning

File issues via GitHub Issues with a clear description

## Commit Messages
Follow Conventional Commits:
```<type>(scope): <short summary>```

Include issue references:
```fixes #42 or refs #105```

## Code Style, Linting & Formatting

Follow Python PEP-8 conventions. No automatic check for style at this point

## Testing

Requires Unit test on each python submodule
Requires an Integration test
Use pytest for both testing modules
Start creating tests once the pipeline can be run through

## Pull Requests & Reviews

Outline PR requirements (template, checklist, size limits), reviewer expectations, approval rules, and required status checks.

PR Template includes: summary of changes, related issue ID, test results (screenshot/logs)
Rules: 
One review from another team member required before merge
Ideally 400 or less lines per PR
Major changes are run by Nicole


## CI/CD

Does not exist at this point. Will automate unit tests and integration tests in the future.

## Security & Secrets

State how to report vulnerabilities, prohibited patterns (hard-coded secrets), dependency update policy, and scanning tools.

## Documentation Expectations

For each new added feature, flag and script: README.md is updated if needed, /docs directory for technical documentation, CHANGELOG.md is updated with new version header and date of last change. 

How this aligns with DOD expectations: README reflects any setup/run instructions required, /docs exists and is updated, Code includes accurate docstrings, changelog records the version and date of change

## Release Process

Versioning: Semantic (major.minor.patch)
Tagging Example: (git tag -a v1.0.0 -m “Initial baseline visualization release”)
(git push origin v1.0.0)
Changelog: generated from commits
Publishing: push to main after team/instructor approval
Rollback: revert tag and reset the main to stable commit


## Support & Contact

All members are expected to respond within 24 hours

Gordon Shum - CLAP Model Lead (shumt@oregonstate.edu)
Henry James - LLM Lead (jameshe@oregonstate.edu)
Lydia Brown - Visualization Lead (browlydi@oregonstate.edu)

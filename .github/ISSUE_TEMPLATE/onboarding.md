---
name: Onboarding
about: Start your Matrix onboarding
title: 'Onboarding for <firstname> <lastname>'
labels: onboarding
assignees: ''

---

**Onboarding**

This issue template provides a structured, technical onboarding checklist for new team members. 

## Steps to be completed by new team members

- [ ] [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repository
- [ ] [Read the onboarding guide](https://docs.dev.everycure.org/onboarding/)
  - [ ] [Install](https://docs.dev.everycure.org/onboarding/installation/) and verify required OS tooling
  - [ ] Validate [pre-commit](https://pre-commit.com/) hooks are working on `git commit`
  - [ ] Setup [Git-Crypt](https://docs.dev.everycure.org/onboarding/git-crypt/#additional-reading) and share GPG public key in the issue
  - [ ] Validate `git-crypt unlock` successfull after key was added by codeowner
  - [ ] Read up on [Kedro](https://docs.dev.everycure.org/onboarding/kedro/)
  - [ ] Read up on [pipeline stages](https://docs.dev.everycure.org/onboarding/pipeline/)
  - [ ] Launched [local setup](https://docs.dev.everycure.org/onboarding/local-setup/) and run test pipeline till completion

**Contributing**

If you spot any issues in the [onboarding guide](https://docs.dev.everycure.org/onboarding/), please submit a pull-request (PR) to improve the experience for future team members.

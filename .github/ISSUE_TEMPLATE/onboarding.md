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
  - [ ] Setup [Git-Crypt](https://docs.dev.everycure.org/onboarding/git-crypt/#additional-reading) and share GPG public key in the issue
  - [ ] Validate `git-crypt unlock` successfull after key was added by codeowner
  - [ ] Read up on [Kedro](https://docs.dev.everycure.org/onboarding/kedro/)
  - [ ] Read up on [pipeline stages](https://docs.dev.everycure.org/onboarding/pipeline/)
  - [ ] Create `.env` file from the `.env.tmpl` file for non docker-based local runs
  - [ ] Launched [local setup](https://docs.dev.everycure.org/onboarding/local-setup/) and run test pipeline till completion


### For Engineers
- [ ] Read the [Infrastructure Introduction Page](https://docs.dev.everycure.org/infrastructure/)

### For (Data) Scientists

- [ ] Watch the [evaluation metrics video by Alexei](https://drive.google.com/file/d/1MYg06oWBIs3LnxrjdZVn_vLJioOGy-Mb/view?usp=share_link)



**Contributing**

If you spot any issues in the [onboarding guide](https://docs.dev.everycure.org/onboarding/), please submit a pull-request (PR) to improve the experience for future team members.

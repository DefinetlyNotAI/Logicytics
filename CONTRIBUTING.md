# Contributing to Logicytics

Looking to contribute something to Logicytics? **Here's how you can help.**

Please take a moment to review this document to make the contribution
process easy and effective for everyone involved.

Following these guidelines helps to communicate that you respect the time of
the developers managing and developing this open source project. In return,
they should reciprocate that respect in addressing your issue or assessing
patches and features.

## Using the issue tracker

The [issue tracker](https://github.com/DefinetlyNotAI/Logicytics/issues) is
the preferred channel for bug reports and features requests
and submitting pull requests, but please respect the following
restrictions:

- Please **do not** derail or troll issues. Keep the discussion on topic and
  respect the opinions of others.

- Please **do not** post comments consisting solely of "+1" or "👍 ".
  Use [GitHub's "reactions" feature](https://blog.github.com/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/)
  instead. We reserve the right to delete comments which violate this rule.

## Issues assignment

I will be looking at the open issues, analyze them, and provide guidance on how to proceed.
Issues can be assigned to anyone other than me** and contributors are welcome
to participate in the discussion and provide their input on how to best solve the issue,
and even submit a PR if they want to.
Please wait that the issue is ready to be worked on before submitting a PR.
We don't want to waste your time.

Please keep in mind that I am small and have limited resources and am not always able to respond immediately.
I will try to provide feedback as soon as possible, but please be patient.
If you don't get a response immediately,
it doesn't mean that we are ignoring you or that we don't care about your issue or PR.
We will get back to you as soon as we can.

If you decide to pull a PR or fork the project, keep in mind that you should only add/edit the scripts you need to,
leave the Explain.md file and the updating of the structure file to me.

## Guidelines for Modifications 📃

When making modifications to the Logicytics project,
please adhere to the following guidelines to ensure consistency and maintainability:

### Basic Check Functions

- **Limitations on Modifications**: Avoid making extreme modifications to the basic check functions in
  the `Logicytics.py` file. Specifically, refrain from altering the fundamental operations related to flags and file
  access mechanisms.

- **Restrictions**: Do not remove core features and program files, especially example files like `CEC` or code in
  the `sys` or `local_libraries` directories

### Documentation and Credit

- **Comments and Docstrings**: Ensure that all additions and modifications are well-documented through comments and
  docstrings. Your contributions should be easy to understand and use, adhering to proper programming etiquette.

- **Credit**: Properly credit your work in the `CREDIT.md` file, following the credit rules outlined in the
  project's [wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki/2-‐-Contribution-Guidelines#credits). Include your
  name and a brief description of your contributions under the appropriate section.

### Debugging and Proof of Work

- **Debugging**: When modifying existing files, demonstrate thorough debugging efforts. Provide evidence of testing and
  debugging processes to ensure the reliability of your changes.

- **Proof of Work**: Include proof of work with your contributions, showcasing the effectiveness and necessity of your
  modifications or additions.

### Adding Features

- **New Files for New Features**: When adding new features, create a separate file for each feature. Each new file must
  contain at least one function and adhere to the project's print rules as specified in
  the [Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki/2-‐-Contribution-Guidelines#printing-rules). While any
  programming language may be used, adherence to
  the [wiki's](https://github.com/DefinetlyNotAI/Logicytics/wiki/2-‐-Contribution-Guidelines) guidelines is mandatory.

- **Integration**: Do not integrate new features directly into the `Logicytics.py` file, neither to the
  local `Flags_Lists.py` library. Leave the integration process to the project maintainers to ensure cohesive project
  structure.

- **Root Folders**: Refrain from modifying root folders such as `.github` or `.git`. Maintain the integrity of the
  project's directory structure.

### Dependencies

- **Requirement File**: Any new libraries introduced as part of your contributions should be listed in
  the `requirements.txt` file. This ensures that the project's dependencies are accurately tracked and managed.

### Must Do's

- **Final Steps**: Ensure that you have run `Logicytics.py --dev` and completed all steps required given to you

- **CREDIT.md**: Ensure that the `CREDIT.md` file has been properly updated. We respect the credit guidelines in the
  project's [wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki/2-‐-Contribution-Guidelines#credits).

- **WiKi**: Ensure that you have followed the project's structure guidelines found in
  the [Wiki](https://github.com/DefinetlyNotAI/Logicytics/wiki).

- **ReadMe**: When modifying specific code in special directories/subdirectories, read their `ReadMe.md` files

## Issues and labels 🛠️

Our bug tracker utilizes several labels to help organize and identify issues.

For a complete look at our labels, see the [project labels page](https://github.com/DefinetlyNotAI/Logicytics/labels).

## Bug reports 🐛

A bug is a _demonstrable problem_ that is caused by the code in the repository.
Good bug reports are extremely helpful!

Guidelines for bug reports:

1. **Use the GitHub issue search** &mdash; check if the issue has already been
   reported.

2. **Check if the issue has been fixed** &mdash; try to reproduce it using the
   latest `main` (or `version` branch if the issue is about a version) in the repository.

A good bug report shouldn't leave others needing to chase you up for more
information. Please try to be as detailed as possible in your report. What is
your environment? What steps will reproduce the issue? What browser(s) and OS
experience the problem? Do other browsers show the bug differently? What
would you expect to be the outcome? All these details will help people to fix
any potential bugs.

## Feature requests 🚀

Feature requests are welcome. But take a moment to find out whether your idea
fits with the scope and aims of the project. It's up to _you_ to make a strong
case to convince the project's developers of the merits of this feature. Please
provide as much detail and context as possible.

## Coding Standards 👨‍💻

- **Code Style**: Follow the project's existing code style.
- **Commit Messages**: Write clear and descriptive commit messages. Use the imperative mood (e.g., "Add feature" instead
  of "Added feature").
- **Documentation**: Update documentation as necessary to reflect any changes you make.

## Pull requests 📝

Good pull requests—patches, improvements, new features—are a fantastic
help. They should remain focused in scope and avoid containing unrelated
commits.

**Please ask first** before embarking on any **significant** pull request (e.g.
implementing features, refactoring code, porting to a different language),
otherwise you risk spending a lot of time working on something that the
project's developers might not want to merge into the project. For trivial
things, or things that don't require a lot of your time, you can go ahead and
make a PR.

Please adhere to the coding guidelines used throughout the
project (indentation, accurate comments, etc.) and any other requirements
(such as test coverage).

View the WiKi for more information on how to write pull requests.

**IMPORTANT**: By submitting a patch, you agree to allow the project owners to
license your work under the terms of the [MIT License](https://github.com/DefinetlyNotAI/Logicytics/blob/main/LICENSE) (
if it
includes code changes) and under the terms of the
[Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).

## License 📝

By contributing your code, you agree to license your contribution under
the [MIT License](https://github.com/DefinetlyNotAI/Logicytics/blob/main/LICENSE).
By contributing to the documentation, you agree to license your contribution under
the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).

## Communication 🗣️

- **Issues**: Use GitHub issues for bug reports and feature requests. Keep the discussion focused and relevant.
- **Pull Requests**: Use pull requests to propose changes. Be prepared to discuss your changes and address any feedback.

If you have any questions or need further clarification, please feel free to contact us at Nirt_12023@outlook.com.

Thank you for your contributions!

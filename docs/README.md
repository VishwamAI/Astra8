# Astra8
8G broadband development

## Auto-Updation Feature

Astra8 includes an auto-updation feature that keeps the project up-to-date with the latest changes from the '7G_8G_development' branch.

### How it works

- The feature uses GitPython to interact with the repository.
- It checks for updates every hour and before running the main tasks.
- If updates are found, the application automatically restarts to apply the changes.

### Prerequisites

- Git must be installed on your system.
- The project must be in a Git repository with the '7G_8G_development' branch set up.

### Configuration

No additional configuration is required. The auto-updation feature is built into the main script.

### Dependencies

The following dependencies are required for the auto-updation feature:
- GitPython
- schedule

These are included in the `requirements.txt` file.

### Usage

The auto-updation feature runs automatically when you start the main script. No manual intervention is needed.

For more details, refer to the `auto_update()` function in the `main.py` file.

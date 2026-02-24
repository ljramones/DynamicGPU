# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a minimal scaffold with only `.java-version` (`25`) committed. As the codebase grows, use a standard Java layout to keep contributions consistent:
- `src/main/java/` for production code
- `src/test/java/` for unit/integration tests
- `src/main/resources/` for config and static resources
- `docs/` for architecture and design notes

Keep packages feature-oriented (for example, `com.dynamisgpu.compute`), and place shared utilities in clearly named modules rather than generic `utils` buckets.

## Build, Test, and Development Commands
No build tool is configured yet. When adding one, document and keep these commands stable in this file and the README.
- `java -version` verifies local JDK compatibility (should match `.java-version`)
- `git status` checks working tree state before commits

Recommended baseline once tooling is introduced:
- `./mvnw clean verify` or `./gradlew build` for full CI-equivalent checks
- `./mvnw test` or `./gradlew test` for local test runs

## Coding Style & Naming Conventions
Use 4-space indentation, UTF-8 text files, and one public class per file. Follow Java naming conventions:
- Classes/interfaces: `PascalCase`
- Methods/fields: `camelCase`
- Constants: `UPPER_SNAKE_CASE`

Prefer descriptive names (`GpuTaskScheduler`) over abbreviations. If a formatter/linter is added (for example, Spotless or Checkstyle), run it before opening a PR.

## Testing Guidelines
Add tests alongside new behavior under `src/test/java`. Name test classes `*Test` (unit) or `*IT` (integration). Use behavior-focused method names such as `rejectsInvalidKernelConfig()`.

Target meaningful coverage for changed code paths, including error handling and boundary conditions.

## Commit & Pull Request Guidelines
There is no commit history yet, so no existing convention to infer. Use Conventional Commits going forward:
- `feat: add initial kernel dispatch API`
- `fix: guard null device context`

PRs should include:
- Clear summary of what changed and why
- Linked issue (if applicable)
- Test evidence (command and result)
- Notes on config or migration impact

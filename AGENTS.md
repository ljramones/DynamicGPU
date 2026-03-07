# Repository Guidelines

## Project Structure & Module Organization
This repository is a Maven multi-module Java project.
- `dynamis-gpu-api`: public GPU interfaces and shared value/error types (`org.dynamisengine.gpu.api`).
- `dynamis-gpu-vulkan`: LWJGL/Vulkan-backed implementations (`org.dynamisengine.gpu.vulkan`).
- `dynamis-gpu-test`: mock implementations and tests used by other modules.
- `dynamis-gpu-bench`: JMH benchmarks.
- Root `pom.xml`: parent build, dependency management, and plugin versions.

Use standard Maven layout per module: `src/main/java` and `src/test/java`.

## Build, Test, and Development Commands
Run from repository root unless noted.
- `mvn clean verify`: full build + test across all modules.
- `mvn test`: run JUnit tests only.
- `mvn -pl dynamis-gpu-vulkan -am test`: test Vulkan module and required dependencies.
- `mvn -pl dynamis-gpu-bench -am package`: build benchmark artifact.
- `mvn -DskipTests package`: fast packaging when iterating on compile-only changes.

Prerequisite: JDK 21+ (enforced by Maven Enforcer plugin).

## Coding Style & Naming Conventions
- Java package prefix: `org.dynamisengine.gpu`.
- Class/interface names: `UpperCamelCase`; methods/fields: `lowerCamelCase`; constants: `UPPER_SNAKE_CASE`.
- Keep files focused by domain (`buffer`, `memory`, `descriptor`, `sync`, etc.).
- Follow existing formatting in each file; avoid mixing styles in a single file.
- Use descriptive, backend-specific names for implementations (for example, `VulkanMemoryOps`, `MockIndirectCommandBuffer`).

## Testing Guidelines
- Framework: JUnit 5 (`org.junit.jupiter`).
- Test class naming: `*Test` (example: `MockImplementationsTest`).
- Place unit tests in each module’s `src/test/java`.
- For API contracts, prefer fast tests using `dynamis-gpu-test` mocks before adding Vulkan-dependent tests.

## Commit & Pull Request Guidelines
Git history follows Conventional Commits (`feat:`, `chore:`). Continue this pattern.
- Commit messages: imperative, scoped, and concise (example: `feat: add Vulkan frame sync coordinator`).
- PRs should include:
  - what changed and why,
  - impacted modules,
  - test/benchmark commands run and results,
  - linked issue(s) when applicable.
- Keep PRs focused; separate refactors from behavior changes.

## Platform & Configuration Notes
- LWJGL runtime in `dynamis-gpu-vulkan` currently includes `natives-macos-arm64`.
- If developing on another OS/arch, add matching LWJGL native classifiers before running Vulkan paths.

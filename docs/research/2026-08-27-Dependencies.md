# Research — 2026-08-27: Dependency injection

**Week:** 4 · **Curriculum area:** FastAPI async & dependency injection
**Time spent:** 25 min

## What I read/watched
- [Dependency Injection Explained Like You’re 5 (with FastAPI Examples)](https://youtu.be/f270BoTicMA?si=LXNwh9kdw-QqgDI9)

## Key ideas
- Instead of an object/function creating a dependency that it needs, it's retrieved from an external source
    - Essentially, the dependency is already created, so we can focus on business logic instead of creating dependency
    - The dependency can then be passed in as a paramater
- An example is a database service. Instead of building a connection, you inject a pre-created connection 
- Instead of a class/module being responsible for creating dependencies, they're provided for us
- Dependency injection loosens coupling and makes our code more modular, testable, and maintainable
- 

## Questions / things I don't understand yet
- I feel like I understand it, but I'm confused on if this increases efficiency.  

## How this applies to my project
- We aren't constructing SQLAlchemy connections directly within our functions. We're doing it externally using dependencies module
- Makes our code more testable
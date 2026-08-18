# Research — 2026-08-17: Ranking Systems (Elo/Glicko)

**Week:** 2 · **Curriculum area:** Elo/Glicko math
**Time spent:** 30 min

## What I read/watched
- [Ranking Systems: Elo, TrueSkill and Your Own](https://youtu.be/VnOVLBbYlU0?si=7G7nzTaMfriAkhR7)

## Key ideas
- Elo formula: RatingDiff = (Score - Expected) * K-factor
    - Score is 0 for loss, 0.5 for draw, 1 for win
    - Expected is 0 to 1, probability of winning
- The real trick is figuring out what the expected result of a game is
- Elo only works for 1v1
- The K-factor needs to be adjusted for new vs experienced players
- TrueSkill improves upon Elo's idea, and was originally implemented in Xbox Live
    - There's two variables: Average skill & degree of uncertainty
    - The degree of uncertainty goes down as a player becomes more experienced
- TrueSkill is more flexible, and quickly converges to the player's true skill
- Calculations are very complex, but it's very easy to model new players (initial rating of 0)
- Alternative to TrueSkill is the Glicko system, but it's limited to 1v1
- The more complex, the more precise the ranking system is
- Elo has subjectivity mainly due to K-factor, but it's objective outside of that
- Glicko & TrueSkill can model Time Decay by increasing uncertainty (Elo can't do this, so inactivity isn't well modeled for it)
- In games, the matchmaking should be random
- Any items that can be compared can use Elo (e.g Amazon, Google search, etc.)

## Questions / things I don't understand yet


## How this applies to my project
- Glicko can be used as a replacement for TrueSkill, and it quickly converges to a player's true skill
    - It also uses uncertainty to account for inactivity
- Try to incorporate home field advantage
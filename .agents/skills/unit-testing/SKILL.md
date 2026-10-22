---
name: unit-testing
description: >-
  Principles for writing unit tests that catch real bugs and survive refactoring —
  framework-agnostic guidance on deciding what to test, how to structure tests, and
  how to avoid the tests that look like coverage but verify nothing. Use this whenever
  writing new tests, reviewing existing tests, adding tests to existing code, or
  closing out any task that changed behavior. Trigger on mentions of unit tests, test
  quality, TDD, mocking strategy, "what should I test here," brittle or flaky tests,
  test coverage, or a task that produced code without tests. Consult this even when
  the user hasn't used the word "test" — if a change has behavior worth protecting,
  the tests need to be good ones, and a task with worthless tests is not done.
---

# Writing Good Unit Tests

A test suite earns its keep two ways: it catches real regressions, and it lets you
change the code freely without rewriting the tests. Most bad test suites fail at both
at once — they pass when the code is broken and break when the code is fine. Almost
every rule below is downstream of avoiding that.

## The One Rule

**Test behavior, not implementation.** Couple a test to *what* the code does, not
*how* it does it. The litmus test: if you safely change the internals or configuration
without changing any required outcome, the tests should stay green. Tests protect
behavior from regressions; they do not protect declarations from change.

Everything else here is a consequence of this rule.

## The Golden Question

Before writing any test, ask:

1. **What incorrect outcome would a user notice?**
2. **Does the test exercise that outcome, or only inspect how it is declared?**

A "user" is whoever consumes the behavior — an end user, calling module, or API
consumer. Merely being able to inspect metadata does not make that metadata behavior.
If no incorrect outcome can be named, omit the test. If the test only inspects a
declaration, exercise the public behavior instead.

## What "Unit" Actually Means

A unit test is a test that **runs in isolation from other tests** — not a test of a
single class or function. This distinction (Kent Beck's original meaning, clarified by
Ian Cooper) changes everything:

- It does **not** mean "one class per test with all collaborators mocked."
- It **does** mean the test doesn't depend on shared mutable state, test execution
  order, or external resources (real network, real DB, real clock, real filesystem).

A single unit test can exercise several collaborating classes together, as long as it's
self-contained and deterministic. Testing one class in total isolation by mocking
everything around it is the most common way to produce tests that are coupled to
structure and verify nothing. **Test at behavioral boundaries** — the public API of a
module or feature — and let the real collaborators inside that boundary participate.

## Deciding What to Test

Spend your testing budget where bugs hide and where breakage hurts.

**Worth testing:**
- **Logic with branches, edges, and decisions** — calculations, parsing, validation,
  state transitions, anything with `if`/`switch`/loops. This is where bugs live.
- **Boundary and error behavior** — empty input, zero, negative, max, off-by-one,
  the malformed payload, the timeout, the thing that throws. Bugs cluster at edges.
- **Contracts consumers rely on** — call the public interface and assert its documented
  outcomes.
- **Bugs you've fixed** — every fix gets a test that fails on the old code. This is
  the highest-value test you can write, because it's provably catching a real defect.

**Not worth testing:**
- **Trivial code with no logic** — plain getters/setters, pass-through wrappers,
  config objects. A test here only restates the code.
- **The language, the framework, or third-party libraries** — don't test that `map`
  maps or that the ORM saves. Assume your dependencies work; test *your* use of them.
- **Implementation details** — private methods, internal call sequences, the fact that
  function A calls function B. Test these *through* the public behavior that uses them.
- **Generated or boilerplate code** with no hand-written decisions.

If you're unsure whether something's worth a test, return to the Golden Question.

## Test the Effect, Not the Declaration

Test a contract through its public interface. Do not inspect its definition merely to
prove that it has not changed. Schema types, enum lists, ORM options, route tables,
dependency wiring, and config values are mechanisms for producing behavior, even when
they implement a public contract.

Ask what would behave incorrectly if the declaration were wrong, then test that:

- Submit valid and invalid values instead of inspecting configured validators.
- Use a query and assert the required related data is available instead of inspecting
  its population settings.
- Exercise the calculation, serialization, or precision requirement instead of
  asserting a schema field's storage type.

Do not duplicate production configuration as the expected value. When an exported
value is the source of truth, use it to choose cases and independently assert their
behavior. Verify, for example, that configured enum values are accepted and a value
outside that set is rejected; do not pin the enum to a second hardcoded list. A test
needs a behavioral oracle, not a second copy of the implementation.

## How to Structure a Test

**Arrange–Act–Assert.** Set up the world, perform one action, check the result. Keep
the three phases visually distinct. (Given–When–Then is the same shape with different
labels — fine to use, especially for behavior-driven phrasing.)

**One behavior per test, not one assertion per test.** The dogma of a single `assert`
per test leads to fragmented, hard-to-read suites. The real rule: each test verifies
**one logical behavior**. That may take several assertions describing different facets
of the same outcome (status *and* the notification it triggered). What you should *not*
do is test multiple unrelated behaviors in one test — when it fails you won't know which
broke.

**Name tests by behavior and condition, not by method.** A good name reads as a
sentence about what the system does and survives a rename of the method under test.

```
Bad:   test_calculateTotal
       test_process_2
Good:  applies bulk discount when quantity exceeds threshold
       rejects an order when the cart is empty
       retries three times before surfacing a network error
```

When a well-named test fails, the name alone tells you what broke without reading the
body.

**Make tests deterministic.** No dependence on wall-clock time, real randomness, system
locale, network, or test ordering. Inject the clock and the RNG so you can pin them.
Flaky tests train people to ignore red, which defeats the entire point of a suite.

## Mocking — The Decision That Makes or Breaks a Suite

Over-mocking is the single most common cause of tests that pass while the code is
broken. The fix is a clear rule about what to replace and what to keep real.

**Mock only what you can't control or don't own — the true I/O boundary:**
network calls, databases, the filesystem, the system clock, randomness, third-party
services. These are slow, non-deterministic, or have side effects you can't allow in a
test.

**Use real implementations for everything you own** — your own services, domain logic,
utilities, value objects. Mocking your own code couples the test to your internal call
structure and turns refactoring into a test-rewriting chore.

```
Bad — everything mocked, including code you own:
  mockDb     = mock()
  mockLogger = mock()
  mockCache  = mock()
  service    = OrderService(mockDb, mockLogger, mockCache)
  // This tests how OrderService wires its collaborators, not what it does.

Good — real internals, test doubles only at the I/O edge:
  db      = InMemoryDatabase()   // real behavior, no I/O
  logger  = TestLogger()         // real logger writing to a buffer
  cache   = InMemoryCache()      // real cache semantics
  service = OrderService(db, logger, cache)
  // Now the test exercises real collaboration and survives refactors.
```

**Prefer fakes over mocks.** A *fake* is a working lightweight implementation — an
in-memory repository, a test email sender that collects messages in a list. A *mock*
records calls and asserts on them. Fakes let you assert on *outcomes* ("the email
landed in the outbox"); mocks tempt you to assert on *interactions* ("`.send()` was
called once with these args"), which couples the test to implementation.

```
Fake you assert against by outcome:
  class TestEmailSender implements EmailSender {
    sent: Email[] = []
    send(email) { this.sent.push(email) }
  }

  it('notifies the customer when an order ships', () => {
    const sender = new TestEmailSender()
    const orders = new OrderService(db, sender)
    const order  = orders.create(customerOrder)
    orders.ship(order.id)

    // Assert the observable outcome, not that send() was invoked.
    expect(sender.sent).toContainEqual(
      objectContaining({ to: customer.email, subject: stringContaining('shipped') })
    )
  })
```

**The tautology trap:** if you mock a dependency to return `X` and then assert the
function returns `X`, you've tested your mock setup, not your code. This is the most
common worthless test and the easiest to write by accident — watch for it.

## Writing Resilient Assertions

Assert on what matters; ignore what doesn't. Over-specified assertions break on
irrelevant changes and bury the one field that actually mattered.

```
Brittle — pins every field, including internal ones:
  expect(result).toEqual({
    id: '123', status: 'complete',
    timestamp: '2024-01-01T00:00:00Z',
    _internalFlag: true, _private: 'detail'
  })

Resilient — asserts the meaningful guarantees, ignores the rest:
  expect(result).toMatchObject({ status: 'complete' })
  expect(result.id).toEqual(expect.any(String))
  expect(result.timestamp).toBeRecentDate()   // a relationship, not an exact value
```

Rules of thumb:
- **Test relationships, not magic constants.** "createdAt is recent" beats pinning an
  exact timestamp; "discount is 10% of subtotal" beats hardcoding `4.50`.
- **Do not duplicate production policy.** An expected list, mapping, schema, or option
  set copied from production is not an independent oracle.
- **Match partial structure** when you only care about some fields.
- **Don't assert on private/internal fields** — if it's prefixed `_` or not part of the
  contract, leave it out.

## Test Data Builders

Inline fixtures with every field spelled out are noisy and brittle. Use a builder that
fills sensible defaults and lets each test override only the field it cares about — so
the test's *intent* is visible and unrelated fields don't distract.

```
function aUser(overrides = {}) {
  return { id: 'u1', name: 'Test User', role: 'member', active: true, ...overrides }
}

// The test states exactly what's relevant to it and nothing more:
const admin = aUser({ role: 'admin' })
```

This keeps tests readable, makes the relevant variable obvious, and means a schema
change touches one builder instead of fifty tests.

## When Writing Tests for Existing Code (the transcription trap)

Writing tests *after* the code — especially when generating them mechanically — has a
specific, dangerous failure mode: you read what the code does and write tests asserting
exactly that, **including its bugs**. This is transcription, not testing. The suite goes
green and proves nothing.

Guard against it:
- **Derive expectations from the spec/intent, not from the current output.** Ask "what
  *should* this return?" not "what *does* it return?" If you can only answer the second,
  you don't yet understand the behavior well enough to test it.
- **Cover the edges the code might have gotten wrong** — empty, null, boundary, error
  paths — not just the happy path the author already had in mind.
- **For a bug fix, write the failing test first** and watch it fail on the unfixed code.
  A test you never saw fail is a test you can't trust.

## TDD Loop (when writing test-first)

1. **Write a test list** — brainstorm the scenarios before touching code. "What should
   happen when…?" This is the most skipped and most valuable step.
2. **Pick one** and make it a concrete, runnable test.
3. **Watch it fail** — and fail for the *right reason*, not a typo or import error. A
   test that passes immediately is either wrong or testing nothing new.
4. **Write the minimal code** to pass. Resist building ahead.
5. **Watch everything pass.**
6. **Refactor** with the tests green. Interface design happened in step 2; implementation
   design happens here. Keep them separate.
7. **Repeat** until the list is empty.

## Review Checklist

Before considering tests done, run through these:

- [ ] Would behavior-preserving implementation or configuration changes keep tests green?
- [ ] Does each test answer "what would a user notice if this failed?" with something real?
- [ ] Is each contract tested through its interface rather than by inspecting its declaration?
- [ ] Are any expected lists, mappings, schemas, or options duplicated from production?
- [ ] Are mocks limited to I/O boundaries, with real implementations for owned code?
- [ ] Any tautological tests (mock returns X, assert X)? Delete or rewrite them.
- [ ] Are assertions on meaningful guarantees, not over-specified internal structure?
- [ ] Are edge cases and error paths covered, not just the happy path?
- [ ] Does each failing test's *name* tell you what broke?
- [ ] Is every test deterministic — no clock, RNG, network, or ordering dependence?
- [ ] Does every bug fix have a test that fails on the old code?

// Local test for wait-for-jobs action logic.
//
// Runs the single-pass evaluator (one polling iteration) against mocked
// `jobs` arrays and asserts the expected outcome: success | failure | wait.
//
// Usage: node .github/actions/wait-for-jobs/test_local.js

'use strict';

// ---- Evaluator (mirror of the script in action.yml) ---------------------
// Pure function: given a jobs array and specs, return what one iteration
// of the polling loop would decide. No I/O, no sleeping.
function evaluate(jobs, jobSpecs, { cached = false } = {}) {
  const normalizedSpecs = jobSpecs.map(spec => {
    if (typeof spec === 'string') {
      return { prefix: spec, expected_count: 1, exact: true };
    }
    return { ...spec, exact: false };
  });

  const totalExpectedJobs = normalizedSpecs.reduce((s, x) => s + x.expected_count, 0);

  const matchesSpec = (jobName, spec) => {
    if (spec.exact) return jobName === spec.prefix;
    return jobName === spec.prefix || jobName.startsWith(spec.prefix + ' (');
  };

  let allCompleted = true;
  let failedJobs = [];
  let completedCount = 0;
  let totalCount = 0;

  for (const spec of normalizedSpecs) {
    const matchingJobs = jobs.filter(job => matchesSpec(job.name, spec));

    for (const job of matchingJobs) {
      totalCount++;
      if (job.status === 'completed') {
        completedCount++;
        if (job.conclusion !== 'success' && job.conclusion !== 'skipped') {
          failedJobs.push(job.name);
        }
      } else {
        allCompleted = false;
      }
    }

    if (matchingJobs.length < spec.expected_count) {
      const unexpandedSkip = matchingJobs.length === 1 &&
        matchingJobs[0].name === spec.prefix &&
        matchingJobs[0].status === 'completed' &&
        matchingJobs[0].conclusion === 'skipped';
      if (unexpandedSkip) {
        const missing = spec.expected_count - 1;
        totalCount += missing;
        completedCount += missing;
      } else {
        allCompleted = false;
      }
    }
  }

  if (failedJobs.length > 0) return { decision: 'failure', failedJobs, completedCount, totalCount, totalExpectedJobs };
  if (allCompleted && totalCount >= totalExpectedJobs) return { decision: 'success', completedCount, totalCount, totalExpectedJobs };
  return { decision: 'wait', completedCount, totalCount, totalExpectedJobs };
}

// ---- Scenarios ----------------------------------------------------------
const stageASpecs = [
  'stage-a-test-1-gpu-small',
  { prefix: 'stage-a-test-cpu', expected_count: 4 },
];

const mk = (name, status, conclusion) => ({ name, status, conclusion });

const scenarios = [
  {
    name: 'BUG REPRO: stage-a-test-cpu skipped at job level (1 bare entry, expected 4)',
    specs: stageASpecs,
    jobs: [
      mk('stage-a-test-1-gpu-small', 'completed', 'success'),
      mk('stage-a-test-cpu', 'completed', 'skipped'),
    ],
    expect: 'success',
  },
  {
    name: 'All 4 matrix entries ran and succeeded',
    specs: stageASpecs,
    jobs: [
      mk('stage-a-test-1-gpu-small', 'completed', 'success'),
      mk('stage-a-test-cpu (0)', 'completed', 'success'),
      mk('stage-a-test-cpu (1)', 'completed', 'success'),
      mk('stage-a-test-cpu (2)', 'completed', 'success'),
      mk('stage-a-test-cpu (3)', 'completed', 'success'),
    ],
    expect: 'success',
  },
  {
    name: 'One of 4 matrix entries failed',
    specs: stageASpecs,
    jobs: [
      mk('stage-a-test-1-gpu-small', 'completed', 'success'),
      mk('stage-a-test-cpu (0)', 'completed', 'success'),
      mk('stage-a-test-cpu (1)', 'completed', 'failure'),
      mk('stage-a-test-cpu (2)', 'completed', 'success'),
      mk('stage-a-test-cpu (3)', 'completed', 'success'),
    ],
    expect: 'failure',
  },
  {
    name: 'Streaming matrix: only first shard has appeared, expanded name (should WAIT, not false success)',
    specs: stageASpecs,
    jobs: [
      mk('stage-a-test-1-gpu-small', 'completed', 'success'),
      mk('stage-a-test-cpu (0)', 'completed', 'skipped'),
    ],
    expect: 'wait',
  },
  {
    name: 'No matrix entries visible yet (should WAIT)',
    specs: stageASpecs,
    jobs: [
      mk('stage-a-test-1-gpu-small', 'completed', 'success'),
    ],
    expect: 'wait',
  },
  {
    name: 'Matrix still running (mix of in_progress and completed)',
    specs: stageASpecs,
    jobs: [
      mk('stage-a-test-1-gpu-small', 'completed', 'success'),
      mk('stage-a-test-cpu (0)', 'completed', 'success'),
      mk('stage-a-test-cpu (1)', 'in_progress', null),
      mk('stage-a-test-cpu (2)', 'queued', null),
      mk('stage-a-test-cpu (3)', 'queued', null),
    ],
    expect: 'wait',
  },
  {
    name: 'Non-matrix (exact) job skipped: single entry, treated as success via line 144',
    specs: ['stage-a-test-1-gpu-small'],
    jobs: [mk('stage-a-test-1-gpu-small', 'completed', 'skipped')],
    expect: 'success',
  },
  {
    name: 'Stage-b spec shape (real production values)',
    specs: [
      { prefix: 'stage-b-test-1-gpu-small', expected_count: 8 },
      { prefix: 'stage-b-test-1-gpu-large', expected_count: 14 },
      { prefix: 'stage-b-test-2-gpu-large', expected_count: 4 },
      { prefix: 'stage-b-test-4-gpu-b200', expected_count: 1 },
    ],
    jobs: [
      mk('stage-b-test-1-gpu-small', 'completed', 'skipped'),
      mk('stage-b-test-1-gpu-large', 'completed', 'skipped'),
      mk('stage-b-test-2-gpu-large', 'completed', 'skipped'),
      mk('stage-b-test-4-gpu-b200', 'completed', 'skipped'),
    ],
    expect: 'success',
  },
];

// ---- Runner -------------------------------------------------------------
let passed = 0;
let failed = 0;
for (const s of scenarios) {
  const result = evaluate(s.jobs, s.specs);
  const ok = result.decision === s.expect;
  const tag = ok ? 'PASS' : 'FAIL';
  console.log(`[${tag}] ${s.name}`);
  console.log(`       expected=${s.expect}  got=${result.decision}  ${JSON.stringify({ completedCount: result.completedCount, totalCount: result.totalCount, totalExpectedJobs: result.totalExpectedJobs })}`);
  if (result.failedJobs) console.log(`       failedJobs=${JSON.stringify(result.failedJobs)}`);
  ok ? passed++ : failed++;
}
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);

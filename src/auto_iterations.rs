//! Automatic choice of the maximum iterations for the zoom of a view.
//!
//! Start: a limit that grows logarithmically with the magnification, `start + per_doubling * log2(magnification)`.
//! Feedback: a sparse grid of probe points is iterated to that limit. Probes that are certified interior (cardioid / period-two bulb test, or an exactly repeating orbit)
//! are finished. The others, that reached the limit without a certificate, are *unresolved*: they may be inside the set or just slow to escape.
//! The limit is multiplied by `growth` and only the unresolved probes continue from where they were (no work is repeated). If enough of them escape
//! (`min_new_escape_fraction` of all probes), the higher limit uncovered detail and the next round follows; otherwise the extra iterations were not worth it and the previous limit is kept.
//! A large fraction at the limit alone proves nothing, since true interior points always end there; only points that *newly escape* count.
//!
//! Limits of the method: this is a heuristic on a sparse grid. Detail narrower than the probe spacing, or orbits that escape after more iterations than the ceiling, can be missed.
//! The result is deterministic: it depends only on the view, the settings, and the iteration arithmetic, not on timing or thread scheduling.
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crate::complex::Complex;
use crate::complex_plane::ComplexPlane;
use crate::mandelbrot_set::MandelbrotSet;

/// The parameters of the automatic iteration limit: where it starts, how it grows with the zoom, and the bounds of the refinement (see the module documentation).
///
/// The named presets are `AutoPreset::settings`.
#[derive(Clone, Debug)]
pub struct AutoIterationSettings {
    /// Limit at magnification 1
    pub start: u32,
    /// Added limit for every doubling of the magnification
    pub per_doubling: u32,
    /// The limit never goes below this
    pub minimum: u32,
    /// The limit never goes above this, whatever the feedback says
    pub ceiling: u32,
    /// Multiplier of the limit in every refinement round (at least 2)
    pub growth: u32,
    /// At most this many refinement rounds after the initial limit
    pub max_rounds: u32,
    /// A round is worth its limit when at least this fraction of all probes newly escapes in it
    pub min_new_escape_fraction: f64,
    /// Probes lie on a grid with this distance in pixels
    pub probe_stride: usize,
    /// Upper bound of the iteration steps spent by the probes over all rounds
    pub work_budget: u64,
}

impl Default for AutoIterationSettings {
    fn default() -> Self {
        AutoIterationSettings { start: 250, per_doubling: 100, minimum: 100, ceiling: 100_000, growth: 2, max_rounds: 6, min_new_escape_fraction: 0.002, probe_stride: 8, work_budget: 2_000_000_000 }
    }
}

/// Named settings for the automatic iteration limit. None of them guarantees that every slowly escaping pixel is found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoPreset {
    /// Lower limits and fewer, coarser probes: faster renders, more deep boundary detail left unresolved
    Performance,
    /// The default: a limit of 250 at zoom 1 plus 100 per zoom doubling, refined while at least 0.2% of the probes newly escape
    Balanced,
    /// Higher starting limits, finer probes and a lower threshold for refining: resolves more structure, renders can take much longer
    HighDetail,
    /// Settings that were edited by hand and match no preset
    Custom,
}

impl AutoPreset {
    /// The presets that the window offers (`Custom` only appears once settings were edited by hand)
    pub const SELECTABLE: [AutoPreset; 3] = [AutoPreset::Performance, AutoPreset::Balanced, AutoPreset::HighDetail];

    /// The name of the preset as the window shows it
    pub fn name(self) -> &'static str {
        match self {
            AutoPreset::Performance => "Performance",
            AutoPreset::Balanced => "Balanced",
            AutoPreset::HighDetail => "High Detail",
            AutoPreset::Custom => "Custom",
        }
    }

    /// The settings of the preset; `Custom` has none of its own and gives the default
    pub fn settings(self) -> AutoIterationSettings {
        match self {
            AutoPreset::Performance => AutoIterationSettings { start: 150, per_doubling: 60, minimum: 100, ceiling: 20_000, growth: 2, max_rounds: 4, min_new_escape_fraction: 0.005, probe_stride: 12, work_budget: 500_000_000 },
            AutoPreset::Balanced | AutoPreset::Custom => AutoIterationSettings::default(),
            AutoPreset::HighDetail => AutoIterationSettings { start: 500, per_doubling: 150, minimum: 100, ceiling: 200_000, growth: 2, max_rounds: 8, min_new_escape_fraction: 0.0005, probe_stride: 4, work_budget: 8_000_000_000 },
        }
    }
}

impl AutoIterationSettings {
    /// The logarithmic starting limit for a view with the given scale (scale 1 = magnification 1; a smaller scale is a higher magnification)
    pub fn initial_limit(&self, scale: f64) -> u32 {
        let doublings = (1.0 / scale).log2().max(0.0);
        let limit = f64::from(self.start) + f64::from(self.per_doubling) * doublings;
        (limit.round() as u32).clamp(self.minimum.min(self.ceiling), self.ceiling)
    }
}

/// Why the search for a limit stopped
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The last round did not uncover enough new detail; the limit before it is used
    Converged,
    /// Every probe is escaped or certified interior
    NothingUnresolved,
    /// The ceiling of the settings was reached
    Ceiling,
    /// The settings allowed no more refinement rounds
    MaxRounds,
    /// The next round would have gone over the work budget of the settings
    WorkBudget,
}

/// What one round of the search found
#[derive(Clone, Debug)]
pub struct Round {
    /// The limit the probes were iterated to in this round
    pub limit: u32,
    /// Probes that have escaped, in this and the earlier rounds
    pub escaped: usize,
    /// Probes certified to be inside the set (the cardioid and bulb test, or an exactly repeating orbit)
    pub certified: usize,
    /// Probes that are at the limit without a certificate: inside the set, or just slow to escape
    pub unresolved: usize,
    /// Probes that escaped in this round after being unresolved at the previous limit
    pub newly_escaped: usize,
}

/// The outcome of choosing a limit
#[derive(Clone, Debug)]
pub struct AutoIterationResult {
    /// The limit to render with
    pub limit: u32,
    /// The palette scale to use: the limit before any refinement, which only depends on the zoom
    pub palette_scale: u32,
    /// The rounds of the search, the first one being the starting limit
    pub rounds: Vec<Round>,
    /// Why the search stopped
    pub stop: StopReason,
    /// The total amount of iteration steps that the probes took, over all rounds
    pub steps: u64,
}

/// An orbit that can be continued with a higher limit; Brent's cycle detection with exact comparison gives the certificate for periodic orbits
struct Probe {
    c: Complex,
    z: (f64, f64),
    iterations: u32,
    saved: (f64, f64),
    since_saved: u32,
    until_save: u32,
    escaped: bool,
}

enum State { Escaped, Certified, Unresolved }

impl Probe {
    /// Creates the orbit of the point `c`, at the start (before the first step)
    fn new(c: Complex) -> Probe {
        Probe { c, z: (0.0, 0.0), iterations: 0, saved: (0.0, 0.0), since_saved: 0, until_save: 1, escaped: false }
    }

    /// Iterates up to `limit` with the same arithmetic as `MandelbrotSet::iterate`; returns the state and the steps taken
    #[allow(clippy::float_cmp)] // An orbit that repeats bit for bit (exact equality) is the certificate that the point never escapes
    fn run(&mut self, m: &MandelbrotSet, limit: u32) -> (State, u64) {
        let r2 = m.orbit_radius * m.orbit_radius;
        let start = self.iterations;
        let (mut zx, mut zy) = self.z;
        let mut state = State::Unresolved;
        while self.iterations < limit {
            let nx = zx * zx - zy * zy + self.c.x;
            let ny = 2.0 * zx * zy + self.c.y;
            (zx, zy) = (nx, ny);
            if nx * nx + ny * ny > r2 {
                state = State::Escaped;
                break;
            }
            self.iterations += 1;
            if zx == self.saved.0 && zy == self.saved.1 {
                state = State::Certified;
                break;
            }
            self.since_saved += 1;
            if self.since_saved == self.until_save {
                self.saved = (zx, zy);
                self.since_saved = 0;
                self.until_save = self.until_save.saturating_mul(2);
            }
        }
        self.z = (zx, zy);
        (state, u64::from(self.iterations - start))
    }
}

/// Chooses the maximum iterations for the view of `c`. Returns `None` if `cancel` was set.
///
/// # Panics
/// If a thread of the probes panics.
pub fn choose_limit(c: &ComplexPlane, m: &MandelbrotSet, width: usize, height: usize, settings: &AutoIterationSettings, threads: usize, cancel: &AtomicBool) -> Option<AutoIterationResult> {
    let initial = settings.initial_limit(c.get_scale());
    let stride = settings.probe_stride.max(1);
    let half = stride as f64 / 2.0;
    let mut probes: Vec<Probe> = Vec::new();
    let mut certified_flags: Vec<bool> = Vec::new();
    for y in (0..height).step_by(stride) {
        for x in (0..width).step_by(stride) {
            let point = c.complex_from_pixel_plane(x as f64 + half, y as f64 + half);
            certified_flags.push(m.is_certified_interior(&point));
            probes.push(Probe::new(point));
        }
    }
    let total = probes.len().max(1);
    let mut rounds: Vec<Round> = Vec::new();
    let mut steps: u64 = 0;
    let mut limit = initial;
    let mut chosen = initial;
    let stop;
    let mut previous_unresolved: Option<usize> = None;
    let mut previous_escaped = 0usize;
    loop {
        //Continue every unresolved probe up to `limit`, in parallel; each probe is independent, so the counts do not depend on the scheduling
        let chunk = probes.len().div_ceil(threads.max(1)).max(1);
        let results: Vec<(usize, usize, usize, u64)> = thread::scope(|scope| {
            let handles: Vec<_> = probes.chunks_mut(chunk).zip(certified_flags.chunks_mut(chunk)).map(|(probes, flags)| {
                scope.spawn(move || {
                    let (mut escaped, mut certified, mut unresolved, mut work) = (0, 0, 0, 0u64);
                    for (probe, flag) in probes.iter_mut().zip(flags.iter_mut()) {
                        if cancel.load(Ordering::Relaxed) { break; }
                        if *flag { certified += 1; continue; }
                        let (state, w) = if probe.escaped { (State::Escaped, 0) } else { probe.run(m, limit) };
                        work += w;
                        match state {
                            State::Escaped => { probe.escaped = true; escaped += 1; }
                            State::Certified => { *flag = true; certified += 1; }
                            State::Unresolved => unresolved += 1,
                        }
                    }
                    (escaped, certified, unresolved, work)
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        if cancel.load(Ordering::Relaxed) { return None; }
        let (escaped, certified, unresolved) = results.iter().fold((0, 0, 0), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.2));
        steps += results.iter().map(|r| r.3).sum::<u64>();
        let newly_escaped = escaped - previous_escaped;
        previous_escaped = escaped;
        rounds.push(Round { limit, escaped, certified, unresolved, newly_escaped });
        //Decide about the round that just finished
        if previous_unresolved.is_some() {
            if (newly_escaped as f64) / (total as f64) >= settings.min_new_escape_fraction {
                chosen = limit;
            } else {
                stop = StopReason::Converged;
                break;
            }
        }
        if unresolved == 0 { stop = StopReason::NothingUnresolved; break; }
        let next = limit.saturating_mul(settings.growth.max(2)).min(settings.ceiling);
        if next <= limit { stop = StopReason::Ceiling; break; }
        if rounds.len() as u32 > settings.max_rounds { stop = StopReason::MaxRounds; break; }
        if steps + (unresolved as u64) * u64::from(next - limit) > settings.work_budget { stop = StopReason::WorkBudget; break; }
        previous_unresolved = Some(unresolved);
        limit = next;
    }
    Some(AutoIterationResult { limit: chosen, palette_scale: initial, rounds, stop, steps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(x: f64, y: f64, scale: f64) -> ComplexPlane {
        let mut c = ComplexPlane::new(160, 120);
        c.set_view(&crate::complex_plane::View::new(x, y, scale));
        c
    }

    fn choose(c: &ComplexPlane, settings: &AutoIterationSettings, threads: usize) -> AutoIterationResult {
        choose_limit(c, &MandelbrotSet::new(1000, 2.0), 160, 120, settings, threads, &AtomicBool::new(false)).unwrap()
    }

    #[test]
    fn presets_are_ordered_from_less_to_more_work_and_balanced_is_the_default() {
        let (p, b, h) = (AutoPreset::Performance.settings(), AutoPreset::Balanced.settings(), AutoPreset::HighDetail.settings());
        assert_eq!(format!("{b:?}"), format!("{:?}", AutoIterationSettings::default()));
        assert!(p.start < b.start && b.start < h.start && p.per_doubling < b.per_doubling && b.per_doubling < h.per_doubling);
        assert!(p.ceiling < b.ceiling && b.ceiling < h.ceiling && p.work_budget < b.work_budget && b.work_budget < h.work_budget);
        assert!(p.min_new_escape_fraction > b.min_new_escape_fraction && b.min_new_escape_fraction > h.min_new_escape_fraction);
        //The starting limits are ordered at every zoom; the final limits also on a boundary view (on smooth views the refinement steps of the presets can land on different multiples)
        for view in [(-0.7498100000000001, -0.020300000000000054, 0.000064), (-0.75, 0.0, 1.0)] {
            let c = plane(view.0, view.1, view.2);
            let results: Vec<AutoIterationResult> = [p.clone(), b.clone(), h.clone()].iter().map(|s| choose(&c, s, 4)).collect();
            assert!(results[0].palette_scale <= results[1].palette_scale && results[1].palette_scale <= results[2].palette_scale);
            assert!(results.iter().all(|r| r.limit >= r.palette_scale));
        }
        let c = plane(-0.7498100000000001, -0.020300000000000054, 0.000064);
        let limits: Vec<u32> = [p, b, h].iter().map(|s| choose(&c, s, 4).limit).collect();
        assert!(limits[0] <= limits[1] && limits[1] <= limits[2], "{limits:?}");
    }

    #[test]
    fn initial_limit_grows_logarithmically_and_respects_the_bounds() {
        let s = AutoIterationSettings { start: 200, per_doubling: 100, minimum: 150, ceiling: 1000, ..Default::default() };
        assert_eq!(s.initial_limit(1.0), 200);
        assert_eq!(s.initial_limit(0.5), 300);
        assert_eq!(s.initial_limit(1.0 / 1024.0), 1000);
        assert_eq!(s.initial_limit(1e-30), 1000, "the ceiling holds");
        assert_eq!(s.initial_limit(100.0), 200, "zooming out never goes below the start");
        let low = AutoIterationSettings { start: 10, minimum: 150, ..s };
        assert_eq!(low.initial_limit(1.0), 150, "the minimum holds");
    }

    #[test]
    fn a_view_inside_the_set_needs_no_refinement_and_a_boundary_view_does() {
        let s = AutoIterationSettings::default();
        let inside = choose(&plane(-0.1, 0.0, 0.05), &s, 4);
        assert_eq!(inside.stop, StopReason::NothingUnresolved);
        assert_eq!(inside.limit, s.initial_limit(0.05), "certified interior points never raise the limit");
        assert_eq!(inside.rounds.len(), 1);
        let boundary = choose(&plane(-0.7498100000000001, -0.020300000000000054, 0.000064), &s, 4);
        assert!(boundary.limit > boundary.palette_scale, "slowly escaping boundary detail raises the limit: {boundary:?}");
        assert!(boundary.rounds.iter().skip(1).any(|round| round.newly_escaped > 0));
    }

    #[test]
    fn cap_hits_alone_do_not_raise_the_limit() {
        //A view of the interior of a period-3 bulb: almost everything ends at the limit, and almost nothing newly escapes
        let result = choose(&plane(-0.12256116687665362, 0.7448617666197442, 0.03), &AutoIterationSettings::default(), 4);
        assert!(result.limit <= 2 * result.palette_scale, "{result:?}");
    }

    #[test]
    fn the_choice_is_deterministic_and_independent_of_the_thread_count() {
        let c = plane(-0.7498100000000001, -0.020300000000000054, 0.000064);
        let s = AutoIterationSettings::default();
        let (a, b, d) = (choose(&c, &s, 1), choose(&c, &s, 4), choose(&c, &s, 7));
        for other in [&b, &d] {
            assert_eq!((a.limit, a.palette_scale, &a.stop, a.steps), (other.limit, other.palette_scale, &other.stop, other.steps));
            assert_eq!(a.rounds.iter().map(|r| (r.limit, r.escaped, r.certified, r.unresolved)).collect::<Vec<_>>(), other.rounds.iter().map(|r| (r.limit, r.escaped, r.certified, r.unresolved)).collect::<Vec<_>>());
        }
    }

    #[test]
    fn the_ceiling_the_rounds_and_the_work_budget_bound_the_refinement() {
        let c = plane(-0.7498100000000001, -0.020300000000000054, 0.000064);
        let capped = choose(&c, &AutoIterationSettings { ceiling: 3000, min_new_escape_fraction: 0.0, ..Default::default() }, 4);
        assert!(capped.limit <= 3000 && capped.rounds.iter().all(|r| r.limit <= 3000), "{capped:?}");
        let few = choose(&c, &AutoIterationSettings { max_rounds: 1, min_new_escape_fraction: 0.0, ..Default::default() }, 4);
        assert!(few.rounds.len() <= 2 && few.stop == StopReason::MaxRounds, "{few:?}");
        let budget = choose(&c, &AutoIterationSettings { work_budget: 200_000, min_new_escape_fraction: 0.0, ..Default::default() }, 4);
        assert_eq!(budget.stop, StopReason::WorkBudget);
    }

    #[test]
    fn cancelling_stops_the_choice() {
        let c = plane(-0.7498100000000001, -0.020300000000000054, 0.000064);
        let cancel = AtomicBool::new(true);
        assert!(choose_limit(&c, &MandelbrotSet::new(1000, 2.0), 160, 120, &AutoIterationSettings::default(), 4, &cancel).is_none());
    }

    #[test]
    fn continuing_an_orbit_gives_the_same_result_as_a_fresh_run() {
        let m = MandelbrotSet::new(1000, 2.0);
        let c = Complex::new(-0.7498100000000001 + 3e-5, -0.020300000000000054 - 1e-5);
        let fresh = m.iterate(&c);
        let mut probe = Probe::new(Complex::new(c.x, c.y));
        let mut state = probe.run(&m, 10).0;
        let mut limit = 10;
        while matches!(state, State::Unresolved) && limit < 1000 { limit = (limit * 2).min(1000); state = probe.run(&m, limit).0; }
        match state {
            State::Escaped => assert_eq!(probe.iterations, fresh),
            State::Certified | State::Unresolved => assert_eq!(fresh, 1000),
        }
    }
}

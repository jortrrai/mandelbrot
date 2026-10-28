//! The views that were visited, with Back and Forward.

use crate::complex_plane::ComplexPlane;

/// The most views that are remembered; the oldest are forgotten first
pub(super) const MAX_VIEWS: usize = 200;

/// How close two views must be to count as the same one: the centers differ by less than this share of the width of the view, and the widths by less than this share of the width
const SAME_VIEW: f64 = 1e-9;

/// The views that were visited in order, and which one is shown now. Going somewhere new after going back forgets the views that were ahead, like the history of a browser.
#[derive(Default)]
pub(super) struct ViewHistory {
    /// The planes in the order they were visited
    planes: Vec<ComplexPlane>,
    /// The index in `planes` of the view that is shown now
    index: usize,
}

impl ViewHistory {
    /// Notes that `plane` is shown now. Nothing happens if it is the same view as the one that is shown (a render of the same view with other settings, or after a resize of the window).
    pub fn record(&mut self, plane: &ComplexPlane) {
        if self.planes.get(self.index).is_some_and(|shown| same_view(shown, plane)) {
            return;
        }
        self.planes.truncate(if self.planes.is_empty() { 0 } else { self.index + 1 });
        self.planes.push(plane.clone());
        if self.planes.len() > MAX_VIEWS {
            self.planes.remove(0);
        }
        self.index = self.planes.len() - 1;
    }

    /// Whether there is a view before the one that is shown
    pub fn can_go_back(&self) -> bool {
        self.index > 0
    }

    /// Whether there is a view after the one that is shown (after going back)
    pub fn can_go_forward(&self) -> bool {
        self.index + 1 < self.planes.len()
    }

    /// Steps back and returns the plane to show, if there is one
    pub fn go_back(&mut self) -> Option<ComplexPlane> {
        self.can_go_back().then(|| {
            self.index -= 1;
            self.planes[self.index].clone()
        })
    }

    /// Steps forward and returns the plane to show, if there is one
    pub fn go_forward(&mut self) -> Option<ComplexPlane> {
        self.can_go_forward().then(|| {
            self.index += 1;
            self.planes[self.index].clone()
        })
    }

    /// The number of views that are remembered
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.planes.len()
    }
}

/// Whether two planes show the same view: the same center and the same width on the real axis, up to `SAME_VIEW`
fn same_view(a: &ComplexPlane, b: &ComplexPlane) -> bool {
    let (center_a, center_b) = (a.center(), b.center());
    let width = a.length_x.abs().max(b.length_x.abs());
    (center_a.x - center_b.x).abs() <= SAME_VIEW * width && (center_a.y - center_b.y).abs() <= SAME_VIEW * width && (a.length_x - b.length_x).abs() <= SAME_VIEW * width
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex_plane::View;

    fn plane(x: f64, scale: f64) -> ComplexPlane {
        let mut plane = ComplexPlane::new(300, 200);
        plane.set_view(&View::new(x, 0.1, scale));
        plane
    }

    fn x_of(plane: &ComplexPlane) -> f64 {
        plane.center().x
    }

    #[test]
    fn views_are_visited_in_order_and_back_and_forward_walk_through_them() {
        let mut history = ViewHistory::default();
        assert!(!history.can_go_back() && !history.can_go_forward() && history.go_back().is_none() && history.go_forward().is_none());
        for x in [0.0, 0.1, 0.2] {
            history.record(&plane(x, 1.0));
        }
        assert_eq!((history.len(), history.can_go_back(), history.can_go_forward()), (3, true, false));
        assert!((x_of(&history.go_back().unwrap()) - 0.1).abs() < 1e-12);
        assert!((x_of(&history.go_back().unwrap()) - 0.0).abs() < 1e-12);
        assert!(history.go_back().is_none(), "the first view is the oldest");
        assert!((x_of(&history.go_forward().unwrap()) - 0.1).abs() < 1e-12);
        assert!(history.can_go_forward());
    }

    #[test]
    fn the_same_view_is_not_recorded_twice_and_going_somewhere_new_forgets_what_was_ahead() {
        let mut history = ViewHistory::default();
        history.record(&plane(0.0, 1.0));
        history.record(&plane(0.0, 1.0));
        history.record(&plane(0.0, 1.0).resized(600, 100)); //A window resize shows the same view
        assert_eq!(history.len(), 1);
        history.record(&plane(0.5, 1.0));
        history.record(&plane(0.5, 0.5));
        history.go_back().unwrap();
        history.record(&plane(0.5, 1.0)); //Rendering the view that is shown again changes nothing
        assert_eq!((history.len(), history.can_go_forward()), (3, true));
        history.record(&plane(0.9, 1.0)); //Somewhere new: the zoomed view that was ahead is forgotten
        assert_eq!((history.len(), history.can_go_forward(), history.can_go_back()), (3, false, true));
        assert!((x_of(&history.go_back().unwrap()) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn only_the_most_recent_views_are_kept() {
        let mut history = ViewHistory::default();
        for step in 0..MAX_VIEWS + 50 {
            history.record(&plane(step as f64 * 0.01, 1.0));
        }
        assert_eq!(history.len(), MAX_VIEWS);
        let mut oldest = 0.0;
        while let Some(plane) = history.go_back() {
            oldest = x_of(&plane);
        }
        assert!((oldest - 0.5).abs() < 1e-9, "the 50 oldest views were forgotten: {oldest}");
    }
}

//! The values that the window uses to translate and zoom the view.

/// How far a move goes and how fast a zoom is
pub struct InteractionVariables {
    ///Variable determining the amount of rows and columns are translated by pressing the 4 arrow keys
    pub translation_amount: u8,
    ///Variable denoting the user scaling speed; the lower this value, the more aggressive the zooming will become
    pub scale_denominator: f64,
    /// The numerator of the factor by which a zoom step scales the plane (see `scaling_factor`)
    pub scale_numerator: f64,
}

impl InteractionVariables{
    /// Creates the variables: the step of a move in pixels, and the numerator and denominator of the zoom factor
    pub fn new(translation_amount: u8, scale_numerator: f64, scale_denominator: f64) -> InteractionVariables {
        InteractionVariables { translation_amount, scale_denominator, scale_numerator }
    }

    /// The factor that zooming in multiplies the plane with: `scale_numerator / scale_denominator`
    pub fn scaling_factor(&self) -> f64 {
        self.scale_numerator / self.scale_denominator
    }

    /// The factor that zooming out multiplies the plane with: the inverse of `scaling_factor`
    pub fn inverse_scaling_factor(&self) -> f64 {
        self.scale_denominator / self.scale_numerator
    }
}

impl Default for InteractionVariables{
    fn default() -> Self {
        InteractionVariables { translation_amount:10, scale_numerator: 9.0, scale_denominator: 10.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_interaction_variables_start_with_the_values_of_the_window() {
        let variables = InteractionVariables::default();
        assert_eq!((variables.translation_amount, variables.scale_numerator, variables.scale_denominator), (10, 9.0, 10.0));
        assert_eq!(variables.scaling_factor(), 0.9, "zooming in multiplies the plane by 9/10");
        assert_eq!(variables.inverse_scaling_factor(), 10.0 / 9.0);
        let faster = InteractionVariables::new(3, 1.0, 10.0);
        assert_eq!((faster.translation_amount, faster.scaling_factor()), (3, 0.1));
    }
}


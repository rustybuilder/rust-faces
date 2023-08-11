use std::fmt::Display;

/// Rectangle.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    /// X coordinate of the top-left corner.
    pub x: f32,
    /// Y coordinate of the top-left corner.
    pub y: f32,
    /// Width of the rectangle.
    pub width: f32,
    /// Height of the rectangle.
    pub height: f32,
}

/// Rectangle position used for chaining constructors.
pub struct RectPosition {
    pub x: f32,
    pub y: f32,
}

impl RectPosition {
    /// Makes a rectangle with the given size.
    pub fn with_size(&self, width: f32, height: f32) -> Rect {
        Rect {
            x: self.x,
            y: self.y,
            width,
            height,
        }
    }

    /// Makes a rectangle with the given end point.
    pub fn ending_at(&self, x: f32, y: f32) -> Rect {
        Rect {
            x: self.x,
            y: self.y,
            width: x - self.x,
            height: y - self.y,
        }
    }
}

impl Rect {
    /// Starts a rectangle with the given position.
    pub fn at(x: f32, y: f32) -> RectPosition {
        RectPosition { x, y }
    }

    /// Right end of the rectangle.
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Bottom end of the rectangle.
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Unites two rectangles.
    ///
    /// # Arguments
    ///
    /// * `other` - Other rectangle to unite with.
    ///
    /// # Returns
    ///
    /// * `Rect` - United rectangle.
    pub fn union(&self, other: &Rect) -> Rect {
        let left = self.x.min(other.x);
        let right = self.right().max(other.right());
        let top = self.y.min(other.y);
        let bottom = self.bottom().max(other.bottom());

        Rect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        }
    }

    /// Intersects two rectangles.
    ///
    /// # Arguments
    ///
    /// * `other` - Other rectangle to intersect with.
    ///
    /// # Returns
    ///
    /// * `Rect` - Intersected rectangle.
    pub fn intersection(&self, other: &Rect) -> Rect {
        let left = self.x.max(other.x);
        let right = self.right().min(other.right());
        let top = self.y.max(other.y);
        let bottom = self.bottom().min(other.bottom());

        Rect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        }
    }

    /// Clamps the rectangle to the given rect.
    /// If the rectangle is larger than the given size, it will be shrunk.
    ///
    /// # Arguments
    ///
    /// * `width` - Width to clamp to.
    /// * `height` - Height to clamp to.
    pub fn clamp(&self, width: f32, height: f32) -> Rect {
        let left = self.x.max(0.0);
        let right = self.right().min(width);
        let top = self.y.max(0.0);
        let bottom = self.bottom().min(height);

        Rect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        }
    }

    /// Calculates the intersection over union of two rectangles.
    ///
    /// # Arguments
    ///
    /// * `other` - Other rectangle to calculate the intersection over union with.
    ///
    /// # Returns
    ///
    /// * `f32` - Intersection over union.
    pub fn iou(&self, other: &Rect) -> f32 {
        let left = self.x.max(other.x);
        let right = (self.right()).min(other.right());
        let top = self.y.max(other.y);
        let bottom = (self.bottom()).min(other.bottom());

        let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
        let area_self = self.width * self.height;
        let area_other = other.width * other.height;

        intersection / (area_self + area_other - intersection)
    }

    /// Calculates the intersection over union of two rectangles.
    ///
    /// # Arguments
    ///
    /// * `other` - Other rectangle to calculate the intersection over union with.
    ///
    /// # Returns
    ///
    /// * `f32` - Intersection over union.
    pub fn iou_min(&self, other: &Rect) -> f32 {
        let left = self.x.max(other.x);
        let right = (self.right()).min(other.right());
        let top = self.y.max(other.y);
        let bottom = (self.bottom()).min(other.bottom());

        let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
        let area_self = self.width * self.height;
        let area_other = other.width * other.height;

        intersection / area_self.min(area_other)
    }

    /// Scales the rectangle.
    pub fn scale(&self, x_scale: f32, y_scale: f32) -> Rect {
        Rect {
            x: self.x * x_scale,
            y: self.y * y_scale,
            width: self.width * x_scale,
            height: self.height * y_scale,
        }
    }

    /// Gets the rectangle as a tuple of (x, y, width, height).
    pub fn to_xywh(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.width, self.height)
    }
}

impl Display for Rect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{x: {}, y: {}, width: {}, height: {}}}",
            self.x, self.y, self.width, self.height
        )
    }
}

use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct RectPosition {
    pub x: f32,
    pub y: f32,
}

impl RectPosition {
    pub fn with_size(&self, width: f32, height: f32) -> Rect {
        Rect {
            x: self.x,
            y: self.y,
            width,
            height,
        }
    }

    pub fn with_end(&self, x: f32, y: f32) -> Rect {
        Rect {
            x: self.x,
            y: self.y,
            width: x - self.x,
            height: y - self.y,
        }
    }
}

impl Rect {
    pub fn at(x: f32, y: f32) -> RectPosition {
        RectPosition { x, y }
    }

    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

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

    pub fn intersection(&self, other: &Rect) -> Rect {
        let left = self.x.max(other.x);
        let right = self.right().min(other.right());
        let top = self.y.max(other.y);
        let bottom = (self.bottom()).min(other.bottom());

        Rect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        }
    }

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

    pub fn scale2d(&self, x_scale: f32, y_scale: f32) -> Rect {
        Rect {
            x: self.x * x_scale,
            y: self.y * y_scale,
            width: self.width * x_scale,
            height: self.height * y_scale,
        }
    }

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

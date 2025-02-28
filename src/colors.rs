pub type Vec3f = (f32, f32, f32);

pub fn rgb_to_hsl((r, g, b): Vec3f) -> Vec3f {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let mut h = (max + min) / 2.0;
    let s;
    let l = h;
    if max == min {
        h = 0.0;
        s = 0.0;
    } else {
        let d = max - min;
        s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };
        h = if max == r {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        h /= 6.0;
    }
    (h * 360.0, s, l)
}

pub fn hsl_to_rgb((h, s, l): Vec3f) -> Vec3f {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (r + m, g + m, b + m)
}

pub fn gamma_correct_encode(v: f32) -> f32 {
    if v * 12.92 <= 0.4045 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub fn gamma_correct_decode(v: f32) -> f32 {
    if v <= 0.4045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

pub fn gamma_correct_encode_rgb((r, g, b): Vec3f) -> Vec3f {
    (
        gamma_correct_encode(r),
        gamma_correct_encode(g),
        gamma_correct_encode(b),
    )
}

pub fn gamma_correct_decode_rgb((r, g, b): Vec3f) -> Vec3f {
    (
        gamma_correct_decode(r),
        gamma_correct_decode(g),
        gamma_correct_decode(b),
    )
}

/// Relative luminance of a color
/// https://www.w3.org/TR/WCAG22/#dfn-relative-luminance
pub fn rgb_to_relative_luminance((r, g, b): Vec3f) -> f32 {
    let (r, g, b) = gamma_correct_decode_rgb((r, g, b));
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

pub fn contrast_ratio(color1: Vec3f, color2: Vec3f) -> f32 {
    let l1 = rgb_to_relative_luminance(color1);
    let l2 = rgb_to_relative_luminance(color2);
    if l1 > l2 {
        (l1 + 0.05) / (l2 + 0.05)
    } else {
        (l2 + 0.05) / (l1 + 0.05)
    }
}

pub fn mix(color1: Vec3f, color2: Vec3f, ratio: f32) -> Vec3f {
    (
        color1.0 * ratio + color2.0 * (1.0 - ratio),
        color1.1 * ratio + color2.1 * (1.0 - ratio),
        color1.2 * ratio + color2.2 * (1.0 - ratio),
    )
}

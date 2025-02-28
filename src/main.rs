mod colors;

use clap::{Parser, ValueEnum};
use colors::Vec3f;

#[derive(Copy, Clone, PartialEq, ValueEnum, Debug)]
pub enum WcagLevel {
    AA,
    AAA,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(index = 1)]
    image: String,
    #[clap(index = 2)]
    output: String,
    #[clap(long, default_value = "5.0")]
    kappa_1: f32,
    #[clap(long, default_value = "10.0")]
    kappa_2: f32,
    #[clap(long, default_value = "0.05")]
    tau: f32,
    #[clap(long, default_value = "aaa")]
    wcag_level: WcagLevel,
    #[clap(long, default_value = "0.1")]
    background_lightness: f32,
}

fn image_to_rgb_data(image: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<Vec3f> {
    let rgb_data: Vec<Vec3f> = image
        .pixels()
        .map(|pixel| {
            let pixel = *pixel;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            (r, g, b)
        })
        .collect();
    rgb_data
}

fn _rgb_data_to_image(
    data: &Vec<Vec3f>,
    dimensions: (u32, u32),
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (width, height) = dimensions;
    let mut image = image::ImageBuffer::new(width, height);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let (r, g, b) = data[(x + y * width) as usize];
        *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
    }
    image
}

fn degree_to_radian(degree: f32) -> f32 {
    degree * std::f32::consts::PI / 180.0
}

fn radian_to_degree(radian: f32) -> f32 {
    radian * 180.0 / std::f32::consts::PI
}

fn i0(x: f32, order: u8) -> f32 {
    let mut sum = 0.0;
    let mut factor = 1.0;
    for i in 0..order {
        let i = i as f32;
        sum += factor * x.powf(i * 2.0);
        factor /= 4.0 * (i + 1.0) * (i + 1.0);
    }
    sum
}

fn get_von_mises_distribution(mu: f32, kappa: f32, order: u8) -> impl Fn(f32) -> f32 {
    let normalization_factor = 1.0 / (2.0 * std::f32::consts::PI * i0(kappa, order));
    move |theta: f32| std::f32::consts::E.powf(kappa * (theta - mu).cos()) * normalization_factor
}

fn weighted_mean(xs: &Vec<f32>, ws: &Vec<f32>) -> f32 {
    let sum: f32 = xs.iter().zip(ws.iter()).map(|(&x, &w)| w * x).sum();
    let mean = sum / ws.iter().sum::<f32>();
    mean
}

fn circular_weighted_mean(xs: &Vec<f32>, ws: &Vec<f32>) -> f32 {
    let sum_sin: f32 = xs.iter().zip(ws.iter()).map(|(&x, &w)| w * x.sin()).sum();
    let sum_cos: f32 = xs.iter().zip(ws.iter()).map(|(&x, &w)| w * x.cos()).sum();
    let mean = sum_sin.atan2(sum_cos);
    if mean < 0.0 {
        mean + std::f32::consts::PI * 2.0
    } else {
        mean
    }
}

fn weighted_softmax(xs: &Vec<f32>, ws: &Vec<f32>, tau: f32) -> Vec<f32> {
    let mut ys: Vec<f32> = xs
        .iter()
        .zip(ws.iter())
        .map(|(&x, &w)| x * w / tau)
        .collect();
    let max = ys.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let sum: f32 = ys.iter().map(|&x| (x - max).exp()).sum();
    ys = ys.iter().map(|&x| (x - max).exp() / sum).collect();
    ys
}

fn get_hue_weights(primary_hue: f32, hues: &Vec<f32>, kappa: f32) -> Vec<f32> {
    let mu = degree_to_radian(primary_hue);
    let von_mises_distribution = get_von_mises_distribution(mu, kappa, 3);
    let ws: Vec<_> = hues
        .iter()
        .map(|&hue| von_mises_distribution(degree_to_radian(hue)))
        .collect();
    ws
}

fn get_chroma_weights(
    primary_hue: f32,
    hues: &Vec<f32>,
    chromas: &Vec<f32>,
    kappa: f32,
) -> Vec<f32> {
    let hue_weights = get_hue_weights(primary_hue, hues, kappa);
    let ws = hue_weights
        .iter()
        .zip(chromas.iter())
        .map(|(&w, &chroma)| w * chroma)
        .collect();
    ws
}

fn get_secondary_hue(primary_hue: f32, hues: &Vec<f32>, chromas: &Vec<f32>, kappa: f32) -> f32 {
    let xs: Vec<_> = hues
        .iter()
        .map(|&degree: &f32| degree_to_radian(degree))
        .collect();
    let ws = get_chroma_weights(primary_hue, hues, chromas, kappa);
    let mean = circular_weighted_mean(&xs, &ws);
    let secondary_hue = radian_to_degree(mean);
    secondary_hue
}

fn get_dark_secondary_hsl(
    secondary_hue: f32,
    hues: &Vec<f32>,
    lightnesses: &Vec<f32>,
    chromas: &Vec<f32>,
    kappa: f32,
    tau: f32,
) -> Vec3f {
    let chroma_weights = get_chroma_weights(secondary_hue, hues, chromas, kappa);
    let ws = weighted_softmax(&lightnesses, &chroma_weights, tau);

    let chroma = weighted_mean(chromas, &ws);
    let lightness = weighted_mean(lightnesses, &ws);
    let saturation = get_saturation_from_chroma(chroma, lightness);
    (secondary_hue, saturation, lightness)
}

fn get_light_secondary_hsl(
    secondary_hue: f32,
    hues: &Vec<f32>,
    lightnesses: &Vec<f32>,
    chromas: &Vec<f32>,
    kappa: f32,
    tau: f32,
) -> Vec3f {
    let chroma_weights = get_chroma_weights(secondary_hue, hues, chromas, kappa);
    let darknesses = lightnesses.iter().map(|&l| 1.0 - l).collect();
    let ws = weighted_softmax(&darknesses, &chroma_weights, tau);

    let chroma = weighted_mean(chromas, &ws);
    let lightness = weighted_mean(lightnesses, &ws);
    let saturation = get_saturation_from_chroma(chroma, lightness);
    (secondary_hue, saturation, lightness)
}

fn get_chroma_from_hsl(hsl: &Vec3f) -> f32 {
    let (_, s, l) = hsl;
    (1.0 - (2.0 * l - 1.0).abs()) * s
}

fn get_saturation_from_chroma(chroma: f32, lightness: f32) -> f32 {
    let mut saturation = chroma / (1.0 - (2.0 * lightness - 1.0).abs());
    saturation = saturation.min(1.0).max(0.0);
    saturation
}

fn binary_search<F>(mut low: f32, mut high: f32, mut f: F) -> f32
where
    F: FnMut(f32) -> bool,
{
    let mut mid = (low + high) / 2.0;
    while high - low > 1e-6 {
        if f(mid) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2.0;
    }
    mid
}

fn find_color_satisfying_wcag_contrast_ratio(
    foreground: Vec3f,
    background: Vec3f,
    target_contrast_ratio: f32,
) -> Vec3f {
    let background_luminance = colors::rgb_to_relative_luminance(background);
    let extreme_foreground = if background_luminance < 0.5 {
        (1.0, 1.0, 1.0)
    } else {
        (0.0, 0.0, 0.0)
    };
    let f = |t: f32| {
        let color = colors::mix(foreground, extreme_foreground, t);
        let contrast_ratio = colors::contrast_ratio(color, background);
        contrast_ratio < target_contrast_ratio
    };
    let t = binary_search(0.0, 1.0, f);
    let color = colors::mix(foreground, extreme_foreground, t);
    color
}

fn main() {
    let args = Args::parse();
    let image_path = args.image;
    let output_path = args.output;
    let kappa_1 = args.kappa_1;
    let kappa_2 = args.kappa_2;
    let tau = args.tau;
    let target_contrast_ratio = if args.wcag_level == WcagLevel::AA {
        4.5
    } else {
        7.0
    };
    let background_lightness = args.background_lightness;

    let get_secondary_hsl = if background_lightness < 0.5 {
        get_dark_secondary_hsl
    } else {
        get_light_secondary_hsl
    };

    dbg!("Reading image...");
    let (width, height) = (1024, 1024);
    let image = image::ImageReader::open(image_path)
        .unwrap()
        .decode()
        .unwrap()
        .resize(width, height, image::imageops::FilterType::Nearest)
        .to_rgb8();
    let (width, height) = image.dimensions();

    dbg!("Generating palette...");
    let rgb_data = image_to_rgb_data(&image);

    let average_color = {
        let color_sum = rgb_data.iter().fold((0.0, 0.0, 0.0), |acc, &rgb| {
            let rgb = colors::gamma_correct_decode_rgb(rgb);
            (acc.0 + rgb.0, acc.1 + rgb.1, acc.2 + rgb.2)
        });
        let n = rgb_data.len() as f32;
        colors::gamma_correct_encode_rgb((color_sum.0 / n, color_sum.1 / n, color_sum.2 / n))
    };
    let background_hsl = colors::rgb_to_hsl(average_color);
    let background_color = {
        let (hue, saturation, _) = background_hsl;
        colors::hsl_to_rgb((hue, saturation, background_lightness))
    };

    let hsl_data: Vec<Vec3f> = rgb_data
        .iter()
        .map(|&rgb| colors::rgb_to_hsl(rgb))
        .collect();

    let hues: Vec<f32> = hsl_data.iter().map(|&(h, _, _)| h).collect();
    let lightnesses: Vec<f32> = hsl_data.iter().map(|&(_, _, l)| l).collect();
    let chromas: Vec<f32> = hsl_data
        .iter()
        .map(|hsl| get_chroma_from_hsl(hsl))
        .collect();

    // R, G, Y, B, M, C
    let primary_hues: Vec<f32> = vec![0.0, 120.0, 60.0, 240.0, 300.0, 180.0];
    let secondary_hues: Vec<f32> = primary_hues
        .iter()
        .map(|&hue| get_secondary_hue(hue, &hues, &chromas, kappa_1))
        .collect();

    let primary_hsls: Vec<Vec3f> = primary_hues.iter().map(|&hue| (hue, 1.0, 0.5)).collect();
    let secondary_hsls: Vec<Vec3f> = secondary_hues
        .iter()
        .map(|&hue| get_secondary_hsl(hue, &hues, &lightnesses, &chromas, kappa_2, tau))
        .collect();

    let primary_colors: Vec<Vec3f> = primary_hsls
        .iter()
        .map(|&hsl: &Vec3f| colors::hsl_to_rgb(hsl))
        .collect();
    let secondary_colors: Vec<Vec3f> = secondary_hsls
        .iter()
        .map(|&hsl: &Vec3f| colors::hsl_to_rgb(hsl))
        .collect();
    let third_colors: Vec<Vec3f> = secondary_colors
        .iter()
        .map(|&color: &Vec3f| {
            find_color_satisfying_wcag_contrast_ratio(
                color,
                background_color,
                target_contrast_ratio,
            )
        })
        .collect();

    let dark_black: Vec3f = (0.0, 0.0, 0.0);
    let bright_black: Vec3f = {
        let (hue, saturation, _) = background_hsl;
        let f = |t: f32| {
            let color = colors::hsl_to_rgb((hue, saturation, t));
            let relative_luminance = colors::rgb_to_relative_luminance(color);
            relative_luminance > 0.25
        };
        let lightness = binary_search(0.0, 1.0, f);
        dbg!(lightness);
        colors::hsl_to_rgb((hue, saturation, lightness))
    };
    let dark_white: Vec3f = {
        let (hue, saturation, _) = background_hsl;
        let f = |t: f32| {
            let color = colors::hsl_to_rgb((hue, saturation, t));
            let relative_luminance = colors::rgb_to_relative_luminance(color);
            dbg!(relative_luminance);
            relative_luminance > 0.50
        };
        let lightness = binary_search(0.0, 1.0, f);
        dbg!(lightness);
        colors::hsl_to_rgb((hue, saturation, lightness))
    };
    let bright_white: Vec3f = (1.0, 1.0, 1.0);

    let dark_colors = &third_colors;
    let bright_colors: Vec<Vec3f> = dark_colors
        .iter()
        .map(|&color: &Vec3f| {
            let (hue, saturation, lightness) = colors::rgb_to_hsl(color);
            let mut lightness = colors::gamma_correct_decode(lightness);
            lightness = colors::gamma_correct_encode(lightness + 0.1);
            colors::hsl_to_rgb((hue, saturation, lightness))
        })
        .collect();

    dbg!(&primary_hues);
    dbg!(&secondary_hues);

    dbg!(&primary_colors);
    dbg!(&secondary_colors);
    dbg!(&third_colors);

    let palette_height = width / primary_colors.len() as u32;
    let output_width = width;
    let output_height = height + palette_height;
    let output_image = image::ImageBuffer::from_fn(output_width, output_height, |x, y| {
        // Image
        if y < height {
            let pixel = *image.get_pixel(x, y);
            return pixel;
        }

        let x = x as f32 / output_width as f32;
        let y = (y - height) as f32 / palette_height as f32;

        let rows = 2;
        let columns = primary_hues.len() + 2;

        let row = (y * rows as f32) as u8;
        let column = (x * columns as f32) as u8;

        let sx = (x - column as f32 / columns as f32) * columns as f32;
        let sy = (y - row as f32 / rows as f32) * rows as f32;

        let (black, colors, white): (Vec3f, &Vec<Vec3f>, Vec3f) = match row {
            0 => (dark_black, &dark_colors, dark_white),
            1 => (bright_black, &bright_colors, bright_white),
            _ => panic!(),
        };
        let foreground_color = match column {
            0 => black,
            1..=6 => colors[(column - 1) as usize],
            7 => white,
            _ => panic!(),
        };

        let rgb = if 0.2 < sx && sx < 0.8 && 0.2 < sy && sy < 0.8 {
            foreground_color
        } else {
            background_color
        };
        image::Rgb([
            (rgb.0 * 255.0) as u8,
            (rgb.1 * 255.0) as u8,
            (rgb.2 * 255.0) as u8,
        ])
    });
    output_image.save(output_path).unwrap();
}

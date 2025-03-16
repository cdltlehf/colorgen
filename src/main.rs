mod colors;

use clap::{Parser, ValueEnum};
use colors::Vec3f;
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Debug, Clone, ValueEnum)]
pub enum WcagLevel {
    AA,
    AAA,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Appearance {
    Darkest,
    Darker,
    Dark,
    Light,
    Lighter,
    Lightest,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(index = 1, conflicts_with = "stdin")]
    image: Option<String>,
    #[clap(long)]
    stdin: bool,
    #[clap(long, default_value = "darker")]
    appearance: Appearance,
    #[clap(long, default_value = "5.0")]
    kappa_1: f32,
    #[clap(long, default_value = "10.0")]
    kappa_2: f32,
    #[clap(long, default_value = "0.05")]
    tau: f32,
    #[clap(long, default_value = "aaa")]
    wcag_level: WcagLevel,
    #[clap(long, default_value = "0.2")]
    minimum_chroma: f32,
    #[clap(long, default_value = "0.2")]
    maximum_background_chroma: f32,
    #[clap(long)]
    debug: bool,
    #[clap(long, requires = "debug")]
    debug_output: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[allow(non_snake_case)]
struct Base16Colors {
    scheme: String,
    author: String,
    base00: String,
    base01: String,
    base02: String,
    base03: String,
    base04: String,
    base05: String,
    base06: String,
    base07: String,
    base08: String,
    base09: String,
    base0A: String,
    base0B: String,
    base0C: String,
    base0D: String,
    base0E: String,
    base0F: String,
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

    if (secondary_hue - primary_hue).abs() > 30.0 {
        primary_hue
    } else {
        secondary_hue
    }
}

fn get_dark_secondary_hsl(
    secondary_hue: f32,
    hues: &Vec<f32>,
    lightnesses: &Vec<f32>,
    chromas: &Vec<f32>,
    kappa: f32,
    tau: f32,
    minimum_chroma: f32,
) -> Vec3f {
    let chroma_weights = get_chroma_weights(secondary_hue, hues, chromas, kappa);
    let ws = weighted_softmax(&lightnesses, &chroma_weights, tau);

    let chroma = weighted_mean(chromas, &ws).max(minimum_chroma);
    let lightness = weighted_mean(lightnesses, &ws)
        .max(minimum_chroma / 2.0)
        .min(1.0 - minimum_chroma / 2.0);
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
    minimum_chroma: f32,
) -> Vec3f {
    let chroma_weights = get_chroma_weights(secondary_hue, hues, chromas, kappa);
    let darknesses = lightnesses.iter().map(|&l| 1.0 - l).collect();
    let ws = weighted_softmax(&darknesses, &chroma_weights, tau);

    let chroma = weighted_mean(chromas, &ws).max(minimum_chroma);
    let lightness = weighted_mean(lightnesses, &ws)
        .max(minimum_chroma / 2.0)
        .min(1.0 - minimum_chroma / 2.0);
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
    minimum_chroma: f32,
    lightmode: bool,
) -> Vec3f {
    let hsl = colors::get_hsl(foreground);

    let extreme_foreground = if lightmode {
        colors::get_color_from_hsl((hsl.0, 1.0, minimum_chroma / 2.0))
    } else {
        colors::get_color_from_hsl((hsl.0, 1.0, 1.0 - minimum_chroma / 2.0))
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

fn set_gamma_correction_encoded_luminance(color: Vec3f, luminance: f32) -> Vec3f {
    let (hue, saturation, _) = colors::get_hsl(color);
    let luminance = colors::gamma_correction_decode(luminance);
    let lightness = colors::find_lightness_for_target_luminance(hue, saturation, luminance);
    colors::get_color_from_hsl((hue, saturation, lightness))
}

fn get_debug_image(image: &image::RgbImage, palette: &Vec<Vec<Vec3f>>) -> image::RgbImage {
    let (width, height) = image.dimensions();
    let palette_height = 128;

    let rows = palette.len() as u8;

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

        let row = (y * rows as f32) as u8;
        let columns = palette[row as usize].len() as u8;
        let column = (x * columns as f32) as u8;

        let color = palette[row as usize][column as usize];
        image::Rgb([
            (color.0 * 255.0) as u8,
            (color.1 * 255.0) as u8,
            (color.2 * 255.0) as u8,
        ])
    });
    output_image
}

fn hue_to_color(
    primary_hue: f32,
    hues: &Vec<f32>,
    lightnesses: &Vec<f32>,
    chromas: &Vec<f32>,
    tau: f32,
    kappa_1: f32,
    kappa_2: f32,
    target_contrast_ratio: f32,
    minimum_chroma: f32,
    background_color: Vec3f,
    lightmode: bool,
) -> Vec3f {
    let get_secondary_hsl = if lightmode {
        get_dark_secondary_hsl
    } else {
        get_light_secondary_hsl
    };

    let secondary_hue = get_secondary_hue(primary_hue, &hues, &chromas, kappa_1);
    let secondary_hsl = get_secondary_hsl(
        secondary_hue,
        &hues,
        &lightnesses,
        &chromas,
        kappa_2,
        tau,
        minimum_chroma,
    );
    let secondary_color = colors::get_color_from_hsl(secondary_hsl);
    let third_color = find_color_satisfying_wcag_contrast_ratio(
        secondary_color,
        background_color,
        target_contrast_ratio,
        minimum_chroma,
        lightmode,
    );
    third_color
}

fn linspace(start: f32, end: f32, n: u32) -> Vec<f32> {
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| start + step * i as f32).collect()
}

fn main() {
    let args = Args::parse();
    let kappa_1 = args.kappa_1;
    let kappa_2 = args.kappa_2;
    let tau = args.tau;
    let target_contrast_ratio = match args.wcag_level {
        WcagLevel::AA => 4.5,
        WcagLevel::AAA => 7.0,
    };
    let minimum_chroma = args.minimum_chroma;
    let maximum_background_chroma = args.maximum_background_chroma;
    let appearance = args.appearance;

    let lightmode = match appearance {
        Appearance::Darkest | Appearance::Darker | Appearance::Dark => false,
        Appearance::Light | Appearance::Lighter | Appearance::Lightest => true,
    };

    let shade_luminances = match appearance {
        Appearance::Darkest => vec![linspace(0.0, 0.9, 6), vec![0.95, 1.0]].concat(),
        Appearance::Darker => vec![linspace(0.1, 0.9, 6), vec![0.95, 1.0]].concat(),
        Appearance::Dark => vec![linspace(0.2, 0.9, 6), vec![0.95, 1.0]].concat(),

        Appearance::Light => vec![linspace(0.8, 0.1, 6), vec![0.05, 0.0]].concat(),
        Appearance::Lighter => vec![linspace(0.9, 0.1, 6), vec![0.05, 0.0]].concat(),
        Appearance::Lightest => vec![linspace(1.0, 0.1, 6), vec![0.05, 0.0]].concat(),
    };

    let (width, height) = (1024, 1024);
    let image = if args.stdin {
        let mut buffer = Vec::new();
        let stdin = std::io::stdin();
        let mut handle = stdin.lock();
        handle.read_to_end(&mut buffer).unwrap();
        image::ImageReader::new(std::io::Cursor::new(buffer))
            .with_guessed_format()
            .unwrap()
            .decode()
    } else {
        image::ImageReader::open(&args.image.clone().unwrap())
            .unwrap()
            .decode()
    }
    .unwrap()
    .resize(width, height, image::imageops::FilterType::Lanczos3)
    .to_rgb8();

    let rgb_data = image_to_rgb_data(&image);

    let average_color = {
        let color_sum = rgb_data.iter().fold((0.0, 0.0, 0.0), |acc, &rgb| {
            let rgb = colors::gamma_correction_decode_rgb(rgb);
            (acc.0 + rgb.0, acc.1 + rgb.1, acc.2 + rgb.2)
        });
        let n = rgb_data.len() as f32;
        colors::gamma_correction_encode_rgb((color_sum.0 / n, color_sum.1 / n, color_sum.2 / n))
    };

    let shade_colors: Vec<Vec3f> = shade_luminances
        .iter()
        .map(|&luminance| {
            set_gamma_correction_encoded_luminance(
                {
                    let (hue, saturation, lightness) = colors::get_hsl(average_color);
                    let saturation = saturation.min(get_saturation_from_chroma(
                        maximum_background_chroma,
                        lightness,
                    ));
                    colors::get_color_from_hsl((hue, saturation, lightness))
                },
                luminance,
            )
        })
        .collect();

    let background_color = shade_colors[0];
    let foreground_color = shade_colors[5];
    let (dark_black, bright_black, dark_white, bright_white) = if lightmode {
        (
            shade_colors[7],
            foreground_color,
            shade_colors[3],
            background_color,
        )
    } else {
        (
            background_color,
            shade_colors[3],
            foreground_color,
            shade_colors[7],
        )
    };

    let hsl_data: Vec<Vec3f> = rgb_data.iter().map(|&rgb| colors::get_hsl(rgb)).collect();
    let hues: Vec<f32> = hsl_data.iter().map(|&(h, _, _)| h).collect();
    let lightnesses: Vec<f32> = hsl_data.iter().map(|&(_, _, l)| l).collect();
    let chromas: Vec<f32> = hsl_data
        .iter()
        .map(|hsl| get_chroma_from_hsl(hsl))
        .collect();

    // R, G, Y, B, M, C
    let primary_hues: Vec<f32> = vec![0.0, 120.0, 60.0, 240.0, 300.0, 180.0];
    let dark_colors: Vec<Vec3f> = primary_hues
        .iter()
        .map(|&hue| {
            hue_to_color(
                hue,
                &hues,
                &lightnesses,
                &chromas,
                tau,
                kappa_1,
                kappa_2,
                target_contrast_ratio,
                minimum_chroma,
                background_color,
                lightmode,
            )
        })
        .collect();

    let bright_colors: Vec<Vec3f> = dark_colors
        .iter()
        .map(|&color: &Vec3f| {
            let (hue, saturation, lightness) = colors::get_hsl(color);
            colors::get_color_from_hsl((hue, saturation, (lightness + 0.1).min(1.0)))
        })
        .collect();

    let (dark_red, dark_green, dark_yellow, dark_blue, dark_magenta, dark_cyan) = (
        dark_colors[0],
        dark_colors[1],
        dark_colors[2],
        dark_colors[3],
        dark_colors[4],
        dark_colors[5],
    );
    let (bright_red, bright_green, bright_yellow, bright_blue, bright_magenta, bright_cyan) = (
        bright_colors[0],
        bright_colors[1],
        bright_colors[2],
        bright_colors[3],
        bright_colors[4],
        bright_colors[5],
    );

    // Base16 Special Colors
    let orange = hue_to_color(
        40.0,
        &hues,
        &lightnesses,
        &chromas,
        tau,
        kappa_1,
        kappa_2,
        target_contrast_ratio,
        minimum_chroma,
        background_color,
        lightmode,
    );
    let brown_luminance = shade_luminances[3];
    let brown = set_gamma_correction_encoded_luminance(orange, brown_luminance);

    let base16_colors = Base16Colors {
        scheme: args.image.clone().unwrap_or("stdin".to_string()),
        author: "Generated by colorgen".to_string(),
        base00: colors::get_hex(shade_colors[0]),
        base01: colors::get_hex(shade_colors[1]),
        base02: colors::get_hex(shade_colors[2]),
        base03: colors::get_hex(shade_colors[3]),
        base04: colors::get_hex(shade_colors[4]),
        base05: colors::get_hex(shade_colors[5]),
        base06: colors::get_hex(shade_colors[6]),
        base07: colors::get_hex(shade_colors[7]),
        base08: colors::get_hex(dark_red),
        base09: colors::get_hex(orange),
        base0A: colors::get_hex(dark_yellow),
        base0B: colors::get_hex(dark_green),
        base0C: colors::get_hex(dark_cyan),
        base0D: colors::get_hex(dark_blue),
        base0E: colors::get_hex(dark_magenta),
        base0F: colors::get_hex(brown),
    };

    println!("{}", serde_yml::to_string(&base16_colors).unwrap());

    if args.debug {
        let palette = vec![
            vec![background_color, foreground_color],
            vec![
                shade_colors[0],
                shade_colors[1],
                shade_colors[2],
                shade_colors[3],
                shade_colors[4],
                shade_colors[5],
                shade_colors[6],
                shade_colors[7],
                dark_red,
                orange,
                dark_yellow,
                dark_green,
                dark_cyan,
                dark_blue,
                dark_magenta,
                brown,
            ],
            vec![
                dark_black,
                dark_red,
                dark_green,
                dark_yellow,
                dark_blue,
                dark_magenta,
                dark_cyan,
                dark_white,
            ],
            vec![
                bright_black,
                bright_red,
                bright_green,
                bright_yellow,
                bright_blue,
                bright_magenta,
                bright_cyan,
                bright_white,
            ],
        ];
        let debug_image = get_debug_image(&image, &palette);
        let debug_path = match args.debug_output.clone() {
            Some(path) => path,
            None => tempfile::Builder::new()
                .suffix(".png")
                .keep(true)
                .tempfile()
                .unwrap()
                .into_temp_path()
                .to_str()
                .unwrap()
                .to_string(),
        };
        debug_image.save(&debug_path).unwrap();
        if args.debug_output == None {
            open::that(&debug_path).unwrap();
        }
    }
}

mod colors;

use clap::{Parser, ValueEnum};
use clap_stdin::FileOrStdin;
use colors::Vec3f;
use itertools::izip;
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
    #[clap(index = 1)]
    image: FileOrStdin,
    #[clap(long, default_value = "darker")]
    appearance: Appearance,
    /// Kappa parameter of the von Mises distribution for hue. The higher the value, the more
    /// weight will be given to hues concentrated around the primary hues: 0, 30 (orange), 60, 120,
    /// 180, 240, 300
    #[clap(long, default_value = "5.0")]
    kappa_1: f32,
    /// Kappa parameter of the von Mises distribution for saturation and lightness. The higher the value, the more weight will be
    /// given to the saturation and lightness of colors with similar hue.
    #[clap(long, default_value = "10.0")]
    kappa_2: f32,
    /// Tau parameter for the softmax function. The higher the value, the more uniform the weights
    /// will be.
    #[clap(long, default_value = "0.05")]
    tau: f32,
    #[clap(long, default_value = "aaa")]
    wcag_level: WcagLevel,
    #[clap(long, default_value = "0.2")]
    minimum_chroma: f32,
    #[clap(long, default_value = "0.2")]
    maximum_background_chroma: f32,
    #[clap(long, default_value = "30.0")]
    hue_difference_threshold: f32,
    /// Do not use the saturation and lightness of the dominant color in the image if the extracted hue is
    /// too far from the source hue.
    #[clap(long)]
    no_use_dominant_color: bool,
    #[clap(long)]
    debug: bool,
    #[clap(long, requires = "debug")]
    debug_output: Option<String>,
    #[clap(long)]
    verbose: bool,
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
        .map(|&pixel| {
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

fn get_hue_weights(source_hue: f32, hues: &Vec<f32>, kappa: f32) -> Vec<f32> {
    let mu = degree_to_radian(source_hue);
    let von_mises_distribution = get_von_mises_distribution(mu, kappa, 3);
    let ws: Vec<_> = hues
        .iter()
        .map(|&hue| von_mises_distribution(degree_to_radian(hue)))
        .collect();
    ws
}

fn extract_hue_and_weight(
    source_hue: f32,
    hues: &Vec<f32>,
    chromas: &Vec<f32>,
    kappa: f32,
) -> (f32, f32) {
    let xs: Vec<_> = hues
        .iter()
        .map(|&degree: &f32| degree_to_radian(degree))
        .collect();
    let hue_weights = get_hue_weights(source_hue, hues, kappa);
    let ws = hue_weights
        .iter()
        .zip(chromas.iter())
        .map(|(&w, &chroma)| w * chroma)
        .collect();
    let mean = circular_weighted_mean(&xs, &ws);
    let total_weight = ws.iter().sum::<f32>() / (ws.len() as f32);
    let extracted_hue = radian_to_degree(mean);
    (extracted_hue, total_weight)
}

fn extract_saturation_and_lightness(
    extracted_hue: f32,
    hues: &Vec<f32>,
    lightnesses: &Vec<f32>,
    chromas: &Vec<f32>,
    kappa: f32,
    tau: f32,
    minimum_chroma: f32,
    lightmode: bool,
) -> (f32, f32) {
    let hue_and_chroma_weights = {
        let hue_weights = get_hue_weights(extracted_hue, hues, kappa);
        hue_weights
            .iter()
            .zip(chromas)
            .map(|(&w, &chroma)| w * chroma)
            .collect()
    };
    let ws = if lightmode {
        let darknesses = lightnesses.iter().map(|&l| 1.0 - l).collect();
        weighted_softmax(&darknesses, &hue_and_chroma_weights, tau)
    } else {
        weighted_softmax(&lightnesses, &hue_and_chroma_weights, tau)
    };

    let chroma = weighted_mean(chromas, &ws).max(minimum_chroma);
    let lightness = weighted_mean(lightnesses, &ws)
        .max(minimum_chroma / 2.0)
        .min(1.0 - minimum_chroma / 2.0);
    let saturation = get_saturation_from_chroma(chroma, lightness);
    (saturation, lightness)
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
    let use_dominant_color = !args.no_use_dominant_color;

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
    let image = {
        let mut buf = Vec::new();
        let mut reader = args.image.clone().into_reader().unwrap();
        reader.read_to_end(&mut buf).unwrap();
        image::ImageReader::new(std::io::Cursor::new(buf))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .resize(width, height, image::imageops::FilterType::Lanczos3)
            .to_rgb8()
    };

    let rgb_data = image_to_rgb_data(&image);

    let average_color = {
        let color_sum = rgb_data.iter().fold((0.0, 0.0, 0.0), |acc, &rgb| {
            let rgb = colors::gamma_correction_decode_rgb(rgb);
            (acc.0 + rgb.0, acc.1 + rgb.1, acc.2 + rgb.2)
        });
        let n = rgb_data.len() as f32;
        colors::gamma_correction_encode_rgb((color_sum.0 / n, color_sum.1 / n, color_sum.2 / n))
    };

    let shade_colors: Vec<_> = shade_luminances
        .iter()
        .map(|&luminance| {
            let (hue, saturation, lightness) = colors::get_hsl(average_color);
            let saturation = saturation.min(get_saturation_from_chroma(
                maximum_background_chroma,
                lightness,
            ));
            let color = colors::get_color_from_hsl((hue, saturation, lightness));
            set_gamma_correction_encoded_luminance(color, luminance)
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

    let hsl_data: Vec<_> = rgb_data.iter().map(|&rgb| colors::get_hsl(rgb)).collect();
    let hues: Vec<_> = hsl_data.iter().map(|&(h, _, _)| h).collect();
    let lightnesses: Vec<_> = hsl_data.iter().map(|&(_, _, l)| l).collect();
    let chromas: Vec<_> = hsl_data
        .iter()
        .map(|hsl| get_chroma_from_hsl(hsl))
        .collect();

    // R, G, Y, B, M, C
    let source_hues = vec![0.0, 120.0, 60.0, 240.0, 300.0, 180.0];
    if args.verbose {
        eprintln!("Source hues: {:?}", source_hues);
    }

    let (extracted_hues, weights): (Vec<_>, Vec<_>) = source_hues
        .iter()
        .map(|&source_hue| extract_hue_and_weight(source_hue, &hues, &chromas, kappa_1))
        .unzip();
    if args.verbose {
        eprintln!("Extracted hues: {:?}", extracted_hues);
        eprintln!("Weights: {:?}", weights);
    }

    let (&dominant_source_hue, &dominant_hue, &dominant_weight) =
        izip!(source_hues.iter(), extracted_hues.iter(), weights.iter())
            .max_by(|(_, _, w1), (_, _, w2)| w1.partial_cmp(w2).unwrap())
            .unwrap();
    let (dominant_saturation, dominant_lightness) = extract_saturation_and_lightness(
        dominant_hue,
        &hues,
        &lightnesses,
        &chromas,
        kappa_2,
        tau,
        minimum_chroma,
        lightmode,
    );
    let dominant_hue_difference =
        (dominant_hue - dominant_source_hue + 180.0).rem_euclid(360.0) - 180.0;
    let exists_dominant_hue =
        dominant_hue_difference.abs() < args.hue_difference_threshold && dominant_weight > 1e-6;

    if args.verbose {
        if exists_dominant_hue {
            eprintln!("Dominant hue: {}", dominant_hue);
            eprintln!("Dominant weight: {}", dominant_weight);
            eprintln!("Dominant saturation: {}", dominant_saturation);
            eprintln!("Dominant lightness: {}", dominant_lightness);
            eprintln!("Dominant hue difference: {}", dominant_hue_difference);
        } else {
            eprintln!("Dominant hue does not exist");
        }
    }

    let names = vec!["Red", "Green", "Yellow", "Blue", "Magenta", "Cyan"];
    let (refined_hues, use_refined_hues): (Vec<_>, Vec<_>) = izip!(
        names.iter(),
        source_hues.iter(),
        extracted_hues.iter(),
        weights.iter()
    )
    .map(|(&name, &source_hue, &extracted_hue, &weight)| {
        let hue_difference = (extracted_hue - source_hue + 180.0).rem_euclid(360.0) - 180.0;
        if args.verbose {
            eprintln!("Source {} hue: {}", name, source_hue);
            eprintln!("Extracted {} hue: {}", name, extracted_hue);
            eprintln!("Hue difference: {}", hue_difference);
        }

        let use_refined_hue = {
            if hue_difference.abs() < args.hue_difference_threshold {
                if args.verbose {
                    eprintln!(
                        "{} hue difference is smaller than the threshold {}.",
                        name, args.hue_difference_threshold
                    );
                }
                true
            } else if weight < minimum_chroma {
                if args.verbose {
                    eprintln!(
                        "{} hue weight is smaller than the minimum chroma {}.",
                        name, minimum_chroma
                    );
                }
                true
            } else {
                false
            }
        };

        if !use_refined_hue {
            (extracted_hue, use_refined_hue)
        } else {
            let refined_hue = {
                if exists_dominant_hue {
                    (source_hue + dominant_hue_difference).rem_euclid(360.0)
                } else {
                    source_hue
                }
            };
            if args.verbose {
                eprintln!(
                    "Using refined hue {} instead of extracted hue.",
                    refined_hue
                );
            }
            (refined_hue, use_refined_hue)
        }
    })
    .unzip();

    let dark_colors: Vec<_> = izip!(names.iter(), refined_hues.iter(), use_refined_hues.iter())
        .map(|(&name, &refined_hue, &use_refined_hue)| {
            if exists_dominant_hue && use_dominant_color && use_refined_hue {
                let dominant_color = colors::get_color_from_hsl((
                    dominant_hue,
                    dominant_saturation,
                    dominant_lightness,
                ));
                let luminance = colors::get_relative_luminance(dominant_color);
                if args.verbose {
                    eprintln!(
                        "Using dominant saturation {} and lightness {} for {}.",
                        dominant_saturation, dominant_lightness, name
                    );
                }
                let lightness = colors::find_lightness_for_target_luminance(
                    refined_hue,
                    dominant_saturation,
                    luminance,
                );
                (refined_hue, dominant_saturation, lightness)
            } else {
                let (saturation, lightness) = extract_saturation_and_lightness(
                    refined_hue,
                    &hues,
                    &lightnesses,
                    &chromas,
                    kappa_2,
                    tau,
                    minimum_chroma,
                    lightmode,
                );
                (refined_hue, saturation, lightness)
            }
        })
        .map(|hsl| colors::get_color_from_hsl(hsl))
        .map(|color| {
            find_color_satisfying_wcag_contrast_ratio(
                color,
                background_color,
                target_contrast_ratio,
                minimum_chroma,
                lightmode,
            )
        })
        .collect();

    let bright_colors: Vec<_> = dark_colors
        .iter()
        .map(|&color: &Vec3f| {
            let luminance = colors::get_relative_luminance(color);
            let (hue, saturation, _) = colors::get_hsl(color);
            let lightness =
                colors::find_lightness_for_target_luminance(hue, saturation, luminance * 1.1);
            colors::get_color_from_hsl((hue, saturation, lightness.min(1.0)))
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
    let orange = {
        let orange_hue = {
            let red_hue = refined_hues[0];
            let yellow_hue = refined_hues[2];
            let difference = (yellow_hue - red_hue).rem_euclid(360.0);
            let orange_hue = (red_hue + difference * 2.0 / 3.0).rem_euclid(360.0);
            orange_hue
        };
        let (orange_saturation, orange_lightness) = extract_saturation_and_lightness(
            orange_hue,
            &hues,
            &lightnesses,
            &chromas,
            kappa_2,
            tau,
            minimum_chroma,
            lightmode,
        );
        if args.verbose {
            eprintln!("Orange hue: {}", orange_hue);
            eprintln!("Orange saturation: {}", orange_saturation);
            eprintln!("Orange lightness: {}", orange_lightness);
        }
        let orange = colors::get_color_from_hsl((orange_hue, orange_saturation, orange_lightness));
        find_color_satisfying_wcag_contrast_ratio(
            orange,
            background_color,
            target_contrast_ratio,
            minimum_chroma,
            lightmode,
        )
    };
    let brown_luminance = shade_luminances[3];
    let brown = set_gamma_correction_encoded_luminance(orange, brown_luminance);

    let base16_colors = Base16Colors {
        scheme: args.image.filename().to_string(),
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

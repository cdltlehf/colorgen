# ColorGen: [Base16](https://github.com/chriskempson/base16) color scheme generator

A color scheme generator for [Base16](https://github.com/chriskempson/base16).

[It is written in rust
btw](https://www.reddit.com/r/linuxmemes/comments/9xgfxq/why_i_use_arch_btw).

## Screenshots

![Colorgen](./screenshot.png)

- [Source](https://open.spotify.com/track/7GhIk7Il098yCjg4BQjzvb)

## Installation

```sh
cargo install --path .
```

## Usage


```
Usage: colorgen [OPTIONS] <IMAGE>

Arguments:
  <IMAGE>  

Options:
      --kappa-1 <KAPPA_1>
          Kappa parameter of the von Mises distribution for hue. The higher the
          value, the more weight will be given to hues concentrated around the
          primary hues: 0, 30 (orange), 60, 120, 180, 240, 300 [default: 5.0]
      --kappa-2 <KAPPA_2>
          Kappa parameter of the von Mises distribution for saturation and
          lightness. The higher the value, the more weight will be given to the
          saturation and lightness of colors with similar hue [default: 10.0]
      --tau <TAU>
          Tau parameter for the softmax function. The higher the value, the
          more uniform the weights will be [default: 0.05]
      --wcag-level <WCAG_LEVEL>
          [default: aaa] [possible values: aa, aaa]
      --minimum-chroma <MINIMUM_CHROMA>
          [default: 0.2]
      --hue-difference-threshold <HUE_DIFFERENCE_THRESHOLD>
          [default: 30.0]
      --no-use-dominant-color
          Do not use the saturation and lightness of the dominant color in the
          image if the extracted hue is too far from the source hue
      --background-color-source <BACKGROUND_COLOR_SOURCE>
          [default: border] [possible values: border, uniform]
      --appearance <APPEARANCE>
          [default: dark] [possible values: dark, light, auto]
      --maximum-dark-background-gamma-encoded-luminance <MAXIMUM_DARK_BACKGROUND_GAMMA_ENCODED_LUMINANCE>
          [default: 0.1]
      --minimum-light-background-gamma-encoded-luminance <MINIMUM_LIGHT_BACKGROUND_GAMMA_ENCODED_LUMINANCE>
          [default: 0.9]
      --maximum-dark-background-chroma <MAXIMUM_DARK_BACKGROUND_CHROMA>
          [default: 0.05]
      --maximum-light-background-chroma <MAXIMUM_LIGHT_BACKGROUND_CHROMA>
          [default: 0.05]
      --verbose
          
      --debug
          
      --debug-output <DEBUG_OUTPUT>
          
  -h, --help
          Print help
  -V, --version
          Print version
```

## Links

- [GitHub](https://github.com/cdltlehf/colorgen)

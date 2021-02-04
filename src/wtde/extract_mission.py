import fire
import wtde

# obviously this will change or go away
SAMPLE_RESULTS_IMAGE = 'pictures_to_scrape/Screenshot from 2021-01-25 13-54-03.png'

def extract(directory):
    image_list = wtde.validate_input(directory)
    # next call a function to give us which image is which
    category = wtde.determine_category(image_list)
    stats_image = wtde.find_stats_screen(image_list, category.name)
    results_header_img = wtde.header_image(stats_image, category)

    header_str = wtde.header_image_to_text(results_header_img)
    try:
      mission_results = {
          'game_category': category.name,
          'game_mode': wtde.game_mode(header_str),
          'map_name': wtde.map_name(header_str),
          'map_type': wtde.map_type(header_str),
          'w_l': wtde.w_or_l(header_str)}
    except ValueError as e:
        print(e)
        results_header_img.show()

    print(mission_results)

def main():
    fire.Fire(extract)

if __name__ == '__main__':
    main()

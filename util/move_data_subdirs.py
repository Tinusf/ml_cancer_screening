import csv
import os


def read_metadata(metadata_path):
    labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    image_id_to_class = {}
    with open(metadata_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[0] == "lesion_id":
                # This is the very first row, which just described the attributes. So skip this.
                continue
            image_id_to_class[row[1]] = labels.index(row[2])
    return image_id_to_class


def move_images(image_folder, image_id_to_class):
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            file_path = os.path.join(root, name)
            image_id = name[:-4]
            label = str(image_id_to_class[image_id])
            new_path = os.path.join(os.path.join(root, ".."), label)
            print(new_path)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            os.rename(file_path, os.path.join(new_path, name))


def main():
    image_id_to_class = read_metadata("../data/skin/HAM10000_metadata.csv")
    move_images("../data/skin/ham10000_images_part_1", image_id_to_class)

    print(image_id_to_class)


if __name__ == '__main__':
    main()

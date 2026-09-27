from src.dicom.reader import read_dicom, read_dicom_pixels


def main():
    print("Проверка обработки ошибок")
    print()

    test_path = "data/raw/Для теста/несуществующий.dcm"

    try:
        read_dicom(test_path)
        print("ОШИБКА: исключение не возникло")
    except FileNotFoundError as error:
        print(f"FileNotFoundError обработан правильно: {error}")

    try:
        read_dicom_pixels(test_path)
        print("ОШИБКА: исключение не возникло")
    except FileNotFoundError as error:
        print(f"FileNotFoundError для Pixel Data обработан правильно: {error}")

    print()
    print("Проверка завершена.")


if __name__ == "__main__":
    main()
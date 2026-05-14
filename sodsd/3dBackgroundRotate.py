try:
    from sodsd.background_rotation import main
except ModuleNotFoundError:
    from background_rotation import main


if __name__ == "__main__":
    main()

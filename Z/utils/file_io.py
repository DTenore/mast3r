def write_prediction(filename, prediction):
    with open(filename, "a") as file:
        file.write(prediction + "\n")
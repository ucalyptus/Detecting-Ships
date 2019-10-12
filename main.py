import __input__ as inp
import __preparing__ as pr
import __training__ as tr
import __result__ as rs


def main():
    input_data, output_data = inp.data()

    X_train, y_train = pr.prep(input_data, output_data)

    model = tr.train(X_train, y_train)

    path = "./input/scenes/scenes/sfbay_1.png"
    rs.detect(model, path)


if __name__ == "__main__":
    main()

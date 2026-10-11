import uvicorn

from evaluator.service import Settings, create_app


def main() -> None:
    uvicorn.run(create_app(Settings.from_env()), host="0.0.0.0", port=8080, access_log=False)


if __name__ == "__main__":
    main()

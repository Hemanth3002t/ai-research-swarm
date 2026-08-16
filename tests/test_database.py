from database.connection import get_connection


def main():
    connection = get_connection()

    print("PostgreSQL connection successful!")

    connection.close()


if __name__ == "__main__":
    main()
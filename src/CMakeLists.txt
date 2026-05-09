add_subdirectory(engine)

if(BUILD_SERVER)
    add_subdirectory(server)

    add_executable(ccinfer-server
        main.cpp
    )

    target_link_libraries(ccinfer-server
        PRIVATE
            server
            engine
    )
endif()

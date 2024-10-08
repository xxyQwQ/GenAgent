- `DF_Sinus`: The Sinus node calculates the sine of a given angle, which can be specified in either radians or degrees. It also offers the option to compute the arcsine of the input value, allowing for flexibility in trigonometric calculations.
    - Parameters:
        - `value`: The angle for which the sine value is to be calculated. This can be in radians or degrees, based on the 'type_' parameter. Type should be `FLOAT`.
        - `type_`: Specifies the unit of the input angle: 'RAD' for radians or 'DEG' for degrees, affecting how the input value is interpreted before calculation. Type should be `COMBO[STRING]`.
        - `arcSin`: A boolean flag that, when true, changes the operation from sine to arcsine, allowing for inverse trigonometric calculations. Type should be `COMBO[BOOLEAN]`.
    - Inputs:
    - Outputs:
        - `float`: The result of the sine or arcsine calculation, depending on the input parameters. Type should be `FLOAT`.

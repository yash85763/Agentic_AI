# Distance Between Two Points in 3D Space
## Why We Calculate It, What It Means, and How the Formula Is Derived

---

## The central question

A point in 3D space tells us a **location**.

For example:

\[
P=(x_1,y_1,z_1)
\]

could describe the position of a drone, a planet, a game character, a molecule, or a data point.

But when we have two points,

\[
P=(x_1,y_1,z_1), \qquad Q=(x_2,y_2,z_2),
\]

we often need to know something beyond their individual locations:

> **How far apart are they in a straight line?**

That is what distance measures.

A coordinate is an **address**. A distance is the length of the invisible string connecting two addresses.

For example, suppose two drones are at:

- Drone \(P\): \((2,1,3)\)
- Drone \(Q\): \((5,5,15)\)

Their coordinates tell us where each drone is relative to an origin. But distance tells us:

- How much cable would connect them?
- Are they close enough to collide?
- Which one is closer to a target?
- How far must one move directly to reach the other?
- In a data setting, how similar or dissimilar are they?

---

# 1. Before coordinates: measuring distance physically

Humans needed distance long before coordinate geometry existed. Distance mattered for:

- dividing land,
- building structures,
- laying roads,
- navigation,
- surveying fields,
- astronomy,
- architecture,
- trade and travel.

At first, distance was measured directly with ropes, marked sticks, footsteps, or chains.

That works when you can physically visit two locations. But it does not give a reusable mathematical rule for arbitrary points.

For instance, suppose one location is **3 units east** and **4 units north** of another. You could walk 3 units east and then 4 units north, covering 7 units of route. But the direct straight-line connection is shorter.

That distinction between a route with turns and direct separation is the seed of the distance formula.

---

# 2. The first key idea: perpendicular movement does not simply add

Suppose you move:

- \(3\) units east,
- \(4\) units north.

These movements meet at a right angle.

```text
        destination
             â
             |\
          4  | \  direct distance
             |  \
             |___\
          start  3
```

The two legs have lengths \(3\) and \(4\). The direct connection is the hypotenuse of a right triangle.

The direct distance is **not**:

\[
3+4=7.
\]

It is:

\[
\sqrt{3^2+4^2}=5.
\]

So:

- \(7\) is the length of a route that turns a corner.
- \(5\) is the straight-line distance through open space.

This difference matters everywhere: navigation, physics, robotics, games, and machine learning.

---

# 3. The Pythagorean theorem: where the distance rule begins

For a right triangle with perpendicular side lengths \(a\) and \(b\), and direct diagonal length \(c\):

\[
a^2+b^2=c^2.
\]

This is the Pythagorean theorem.

The ancient \(3\)-\(4\)-\(5\) triangle gives:

\[
3^2+4^2=5^2
\]

\[
9+16=25.
\]

Therefore:

\[
c=5.
\]

The relationship was known in ancient mathematics before its later association with Pythagoras. Euclid gave a famous geometric proof in *Elements*, Book I, Proposition 47.

## A geometric area derivation

Take a right triangle whose perpendicular sides are \(a\) and \(b\), and whose hypotenuse is \(c\).

Make four copies and arrange them inside a square of side \(a+b\).

The total area of the large square is:

\[
(a+b)^2.
\]

Inside it, there are:

- four right triangles, each with area \(\frac{ab}{2}\),
- one central square with area \(c^2\).

Therefore:

\[
(a+b)^2=4\left(\frac{ab}{2}\right)+c^2.
\]

Expand the left side:

\[
a^2+2ab+b^2=2ab+c^2.
\]

Subtract \(2ab\) from both sides:

\[
a^2+b^2=c^2.
\]

So the theorem is not a random recipe. It follows from the geometry of right angles and areas.

---

# 4. What was still missing before the distance formula?

The Pythagorean theorem solved right-triangle problems. But geometry was still mostly visual:

- draw a figure,
- construct lines,
- measure lengths,
- prove relationships.

What was missing was a numerical way to represent **any location**.

A point needed an address made of numbers.

That changed with coordinate geometry.

---

# 5. Coordinate geometry: turning locations into numbers

In the 1600s, RenÃ© Descartes and Pierre de Fermat developed what is now called analytic or coordinate geometry.

The key idea was revolutionary:

> A geometric point can be represented by numbers.

On a plane, a point can be written as:

\[
(x,y).
\]

In ordinary 3D space, a point can be written as:

\[
(x,y,z).
\]

The coordinates represent movement along mutually perpendicular axes:

- \(x\): left-right,
- \(y\): forward-backward,
- \(z\): up-down.

This united algebra and geometry. A location became a number-based object; geometry could be calculated instead of only drawn.

Once this happened, a new question became natural:

> Given two coordinate addresses, can we calculate their straight-line separation without drawing a picture?

The answer is the distance formula.

---

# 6. Deriving the 2D distance formula

Let:

\[
P=(x_1,y_1)
\]

and:

\[
Q=(x_2,y_2).
\]

To move from \(P\) to \(Q\):

- the horizontal change is

\[
\Delta x=x_2-x_1,
\]

- the vertical change is

\[
\Delta y=y_2-y_1.
\]

These changes are perpendicular, so they form the two legs of a right triangle.

By the Pythagorean theorem:

\[
d^2=(\Delta x)^2+(\Delta y)^2.
\]

Substitute the coordinate differences:

\[
d^2=(x_2-x_1)^2+(y_2-y_1)^2.
\]

Taking the square root gives the actual distance:

\[
\boxed{
d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
}.
\]

## Why do we subtract first?

Coordinates are positions relative to the origin. To compare two points, we need their **change** in each direction.

For example:

\[
P=(2,4), \qquad Q=(5,8).
\]

Then:

\[
\Delta x=5-2=3,
\]

\[
\Delta y=8-4=4.
\]

The direct distance is:

\[
d=\sqrt{3^2+4^2}=5.
\]

The vector from \(P\) to \(Q\) is:

\[
\overrightarrow{PQ}=(3,4).
\]

This vector keeps both direction and component-wise movement. Its magnitude is the distance:

\[
\left\|\overrightarrow{PQ}\right\|=5.
\]

---

# 7. Extending the idea from 2D to 3D

Now let:

\[
P=(x_1,y_1,z_1)
\]

and:

\[
Q=(x_2,y_2,z_2).
\]

The coordinate changes are:

\[
\Delta x=x_2-x_1,
\]

\[
\Delta y=y_2-y_1,
\]

\[
\Delta z=z_2-z_1.
\]

We derive the formula by using the Pythagorean theorem twice.

## Step 1: Find the horizontal separation

Ignore height for a moment. In the \(xy\)-plane, the horizontal separation is:

\[
r=\sqrt{(\Delta x)^2+(\Delta y)^2}.
\]

## Step 2: Combine that horizontal separation with height

The horizontal distance \(r\) and vertical difference \(\Delta z\) are perpendicular. Therefore:

\[
d^2=r^2+(\Delta z)^2.
\]

But:

\[
r^2=(\Delta x)^2+(\Delta y)^2.
\]

Substitute this into the previous equation:

\[
d^2=(\Delta x)^2+(\Delta y)^2+(\Delta z)^2.
\]

Finally, take the square root:

\[
\boxed{
d=
\sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}
}.
\]

This is the standard 3D Euclidean distance formula.

It is simply the Pythagorean theorem applied twice.

---

# 8. Worked 3D example

Let:

\[
P=(2,1,3)
\]

and:

\[
Q=(5,5,15).
\]

Calculate the change in each direction:

\[
\Delta x=5-2=3,
\]

\[
\Delta y=5-1=4,
\]

\[
\Delta z=15-3=12.
\]

Now substitute into the 3D distance formula:

\[
d=\sqrt{3^2+4^2+12^2}.
\]

\[
d=\sqrt{9+16+144}.
\]

\[
d=\sqrt{169}.
\]

\[
\boxed{d=13}.
\]

There is also an intuitive two-stage view:

1. First find horizontal separation:

\[
\sqrt{3^2+4^2}=5.
\]

2. Then combine the horizontal distance with height difference \(12\):

\[
\sqrt{5^2+12^2}=13.
\]

So the direct cable from \(P\) to \(Q\) would have length \(13\).

---

# 9. What distance tells us, and what it does not

Distance answers:

> **How separated are these two points?**

It does not tell us direction.

For direction, use the displacement vector:

\[
\overrightarrow{PQ}=(x_2-x_1,\;y_2-y_1,\;z_2-z_1).
\]

For only the amount of separation, use the vector magnitude:

\[
\left\|\overrightarrow{PQ}\right\|=
\sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}.
\]

| Concept | Question it answers |
|---|---|
| Point | Where is something? |
| Displacement vector | How do I move from one point to another? |
| Distance | How long is the direct straight connection? |
| Path length | How much route did I actually travel? |

For example, from \((0,0)\) to \((3,4)\):

- A grid route is \(3+4=7\).
- The direct straight-line distance is \(5\).

A taxi constrained by streets may travel 7 units. A bird flying directly may travel 5 units.

---

# 10. Why this formula is important

The 3D distance formula appears across science and engineering.

## Physics

It helps calculate or describe:

- the separation of moving objects,
- spacecraft positions,
- collision distances,
- gravitational interactions,
- electric-field relationships,
- orbital motion.

## Engineering and robotics

It helps a robot determine:

- how far its hand or gripper is from an object,
- whether an object is within reach,
- whether parts may collide,
- how far it must travel.

## Computer graphics and games

It is used for:

- collision detection,
- camera positioning,
- object selection,
- lighting calculations,
- enemy detection ranges,
- sound-volume falloff,
- animation and 3D rendering.

## Medicine, chemistry, and biology

It can measure:

- distances between anatomical landmarks,
- molecular structures,
- protein geometry,
- 3D scan features,
- neuron locations.

## Machine learning and embeddings

A model may represent a sentence, image, product, customer, or document as a long numerical vector.

Distance then becomes a possible measure of similarity:

> If two representations are close, the model may regard them as similar in the representation space.

The 3D formula extends to \(n\) dimensions:

\[
d=
\sqrt{(a_1-b_1)^2+(a_2-b_2)^2+\cdots+(a_n-b_n)^2}.
\]

This is called **Euclidean distance**.

The geometry is unchanged. We simply have more than three perpendicular directions.

---

# 11. The deeper intuition

The 3D formula says:

\[
\text{total squared separation}
=
\text{squared x-separation}
+
\text{squared y-separation}
+
\text{squared z-separation}.
\]

Each axis contributes independently because the axes are perpendicular.

That right-angle property is essential.

If the coordinate directions were tilted instead of perpendicular, this simple formula would need extra terms. In advanced mathematics, the generalized rule for measuring length is described by a **metric**. In physics and differential geometry, a metric tensor describes how distances behave in more complicated spaces.

So the ordinary formula quietly assumes:

1. the axes are perpendicular,
2. each axis uses compatible units,
3. the space is flat in the Euclidean sense,
4. the path of interest is the straight line between points.

---

# 12. When this is not the right notion of distance

The 3D Euclidean formula gives the shortest straight-line distance in flat space. But not every real-world problem uses that kind of distance.

Examples:

- Between two cities on Earth, travel follows the curved surface, not a straight line through the planet.
- In a city, roads, walls, traffic, and one-way streets can make the real route much longer.
- In machine learning, features with different scales may need normalization before Euclidean distance is meaningful.
- In curved spaces, the shortest valid path may be a geodesic rather than a straight Euclidean line.

The formula answers one precise question:

> **What is the shortest straight-line separation between two points in flat space?**

---

# Final takeaway

The 3D distance formula was not invented as an isolated rule to memorize.

It emerged through a sequence of ideas:

1. People needed to measure separation.
2. Right triangles revealed that direct distance is not the same as a route with turns.
3. The Pythagorean theorem explained how perpendicular movements combine.
4. Coordinate geometry represented locations with numbers.
5. Subtracting coordinates gave the movement along each axis.
6. Applying the Pythagorean theorem twice produced the 3D formula.

\[
\boxed{
d=
\sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}
}.
\]

In one sentence:

> The formula measures the straight-line length created when independent perpendicular movements in the \(x\), \(y\), and \(z\) directions are combined.

---

## Suggested next concepts to study

1. Vectors and vector magnitude
2. Dot product and its geometric meaning
3. Unit vectors
4. Distance in \(n\)-dimensional vector spaces
5. Euclidean distance versus Manhattan distance
6. Norms and metrics
7. The geometry of embeddings in machine learning

---

## Historical references

- Euclid, *Elements*, Book I, Proposition 47: Pythagorean theorem proof.
- Encyclopaedia Britannica, âAnalytic Geometryâ: historical background on Descartes, Fermat, and coordinate geometry.
